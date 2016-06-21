require 'nn';
require 'nngraph';
require 'torch';
require 'cutorch';
require 'cunn';
--matio=require 'matio';
torch.manualSeed(123)
function build_network(inputSize,hiddenSize,length)
--Building complete network by connecting hidden units of different neural networks. 
  local ns={inputSize=inputSize,hiddenSize=hiddenSize,length=length}
  ns.initial_h=nn.Identity()()
  ns.inputs=nn.Identity()()
  ns.splitInputs=nn.SplitTable(1)(nn.Reshape(2,10)(ns.inputs))
  local unit={h=ns.initial_h}
  for i=1,length do
    local x=nn.Reshape(10)(nn.SelectTable(i)(ns.splitInputs))
    unit=addUnit(unit.h,x,inputSize,hiddenSize)
  end
  ns.y=nn.Linear(hiddenSize,10)(unit.h)
  local mod=nn.gModule({ns.initial_h,ns.inputs},{ns.y})
  ns.paramsTable,ns.gradParamsTable=mod:parameters()
--Parameter sharing across time.
  for t=2,length do
    ns.paramsTable[2*t-1]:set(ns.paramsTable[1])
    ns.paramsTable[2*t]:set(ns.paramsTable[2])
    ns.gradParamsTable[2*t-1]:set(ns.gradParamsTable[1])
    ns.gradParamsTable[2*t]:set(ns.gradParamsTable[2])
  end
  ns.par,ns.gradPar=mod:getParameters()
  mod.ns=ns
  mod=mod:cuda()
  return mod
end
function addUnit(prev_h,x,inputSize,hiddenSize)
--Adding neural network with linear layer followed by non linearity provided by Tanh.
  local ns={}
  ns.phx=nn.JoinTable(1,2)({prev_h,x})
  ns.h=nn.Tanh()(nn.Linear(inputSize+hiddenSize,hiddenSize)({ns.phx}))
  return ns
end
function rnnTrainer(module,criterion,train_length)
--Traning RNN where weights are initialized from gaussian distribution.
  local trainer={}
  trainer.learningRate=0.01
  trainer.module=module
  trainer.criterion=criterion
  trainer.train_length=train_length
  function trainer:train(dataset)
    local currentLearningRate=self.learningRate
    local length=self.train_length
    local module=self.module
    local criterion=self.criterion
    local shuffledIndices=torch.randperm(500,'torch.LongTensor')
    local shuffledIndicesCuda=shuffledIndices:cuda()
    local par=module.ns.par
    local gradPar=module.ns.gradPar
    local iterations=1000
    par:normal(-0.02,0.02)
    for i=1,iterations do
      for t=1,length do
        local example=dataset[shuffledIndicesCuda[t]]
        local input=example[1]
        local target=example[2]
        gradPar:zero()
        op=module:forward(input)
        criterion:forward(module.output,target)
        criterion:backward(module.output,target)
        module:backward(input,criterionCuda.gradInput)
        --par:add(-self.learningRate, gradPar)
        module:updateParameters(0.01)
      end
    end
    return module  
  end
  return trainer
end

function predict(data,module,length,criterion)
--Predicting output on test data.	
  for i=1,length do
    local example=data[i]
    local input=example[1]
    local target=example[2]
    predicted_op=module:forward(input)
    error=criterion:forward(predicted_op,target)
    print(error)
    print 'predicted op is'
    print(predicted_op)
    print 'actual op is'
    print(target)
  end
end
 
function manual_predict(data,module)
--Manual debugging not of use.
  param,gradparam=module:parameters()
  local example=data[1]
  local input=example[1]
  local x=input[2]
  local target=example[2]
  print(target)
  local prev_h=input[1]
  local appended=torch.CudaTensor(1)
  local k=1
  local j=1
  while j<=18 do
    appended[1]=x[1][k]
    final_ip=torch.cat(prev_h,appended,2)
    prev_h=final_ip*(param[j]:t())
    op=prev_h+param[j+1]
    m=nn.Tanh()
    op=m:forward(op)
    j=j+2
    k=k+1
  end
  final_wt=param[19]
  final_op=op*(final_wt:t())+param[20]
  print(final_op)
end

function tablelength(T)
--Calculating length of table
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end 

function loading_data(file)
--Creating data 
  local f = io.open(file, "rb")
  local lines={}
  for line in io.lines(file) do
    lines[#lines + 1] = line
  end
  local rows=tablelength(lines)
  local data=torch.CudaTensor(rows,10)
  for i=1,rows do
    local k=1
    each_line=lines[i]:split(' ')
    local cols=tablelength(each_line)
    for j=1,cols do
      if each_line[j]~="" then
        data[i][k]=tonumber(each_line[j])
        k=k+1
      end
    end 
  end
  return data
end
function preprocess_data()
--Performing preprocessing of data by dividing each dimension by standard deviation 
  local std=torch.CudaTensor(data:size()[2])
  for i=1,data:size()[2] do
    std[i]=data[{{},i}]:std()
  end
  for i=1,data:size()[2] do
    data[{{},i}]:div(std[i])
  end
  
end
function create_trainData()
--Creating training dataset
  local inputs=data_aug[{{1,500},{}}]
  local target=data[{{3,502},{1,10}}]
  local train_data={}
  for i=1,train_length do
    train_data[i]={{initial_hCuda,inputs[i]},target[i]}
  end
  return train_data
end
function create_testData()
--Creating test dataset
  local test_data={}
  local test_input=data_aug[{{501,559},{}}]
  local test_target=data[{{503,561},{1,10}}]
  for i=1,test_length do
    test_data[i]={{initial_hCuda,test_input[i]},test_target[i]}
  end
  return test_data
end
--Loading data
myfile='/gpuusers/ms/cs15s002/RNN/data1.txt'
data=loading_data(myfile)
print 'loading data finished'
preprocess_data()
print 'preprocessing done'
initial_h = torch.zeros(5)
initial_hCuda=initial_h:cuda()
data_aug=torch.CudaTensor(data:size()[1]-2,20)
for i=2,data:size()[1]-1 do
  data_aug[i-1]=torch.cat(data[i-1],data[i],1)
end

rnn=build_network(10,5,2)
print 'RNN Network is built'
criterion=nn.MSECriterion()
criterionCuda=criterion:cuda()
train_length=500
test_length=59
train_data=create_trainData()
print 'Training dataset created'
trainer=rnnTrainer(rnn,criterionCuda,train_length)
module=trainer:train(train_data)
print 'Trainig for 1000 epochs done'
test_data=create_testData()
print 'Test dataset created'
predict(test_data,module,test_length,criterionCuda)
print 'Prediction on test data finished'
--manual_predict(test_data,module)





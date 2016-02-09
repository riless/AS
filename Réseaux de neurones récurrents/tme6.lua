require 'nn'
require 'torch'
require 'nngraph'

-- omp_set_dynamic(0);     -- Explicitly disable dynamic teams
-- omp_set_num_threads(1); -- Use 4 threads for all consecutive parallel regions

local model_utils = require 'model_utils'
--[[
    print(v)
    print(v.x_batches[1])
    print(v.y_batches[1])
]]--
function load_exemples(T) 
    local CharLMMinibatchLoader=require 'as/CharLMMinibatchLoader'
    local v=CharLMMinibatchLoader.create("as/data.t7","as/vocab.t7",1,T)
    return v, v.vocab_size, v.nbatches
end

function number_mapping(v)
	nm = {}
	for key,value in pairs(v.vocab_mapping) do
	  nm[value] = key
	end
	return nm
end

function decode_outputs(v, outputs)
  local string = ''
  for i = 1, #outputs do
    local key = torch.squeeze(v.scalarize(outputs[i])) -- tensor1 to scalar
    local nm = number_mapping(v)
    string = string .. nm[key]
  end
  return string
end


function decode_batch(v, batch)
  local string = ''
  for i = 1, v.seq_length do
    local key = batch[{1,i}]
    local nm = number_mapping(v)
    string = string .. nm[key]
  end  
  return string
end


function char2vec(v, k)
  local vec = torch.zeros(v.vocab_size)
  vec[k] = 1
  return vec
end

---------------[[ ETAPE 1 ]]---------------
-- dim_x taille de mon vocabulaire
-- dim_h taille de l'espace latent


-- retourne la proba
function create_g(dim_x, dim_h)
    local input_x = nn.Identity()()
    local lx = nn.Linear(dim_h, dim_x)(input_x)
    local ln = nn.LogSoftMax(dim_x, dim_x)(lx)
    return nn.gModule({input_x}, {ln})
end



---------------[[ ETAPE 2 ]]---------------

-- retourne les états
function create_h(dim_x, dim_h) -- mémoire ( état des neurones après t mots )
    local input_x = nn.Identity()()
    local input_h = nn.Identity()()

    local lx = nn.Linear(dim_x, dim_h)(input_x)
    local lh = nn.Linear(dim_h, dim_h)(input_h)

    local res = nn.CAddTable()({lx, lh})
    local ln = nn.Tanh(dim_h, dim_h)(res)

    return nn.gModule({input_h, input_x}, {ln})
end




---------------[[ ETAPE 3 ]]---------------
-- Cloner g_theta et h_tau

---------------[[ ETAPE 4 ]]---------------
-- Créer le module RNN


-- h retourne l'etat ( entrée proba et etat précédent )
-- g retourne la proba
function create_rnn(gs, hs, T)

	local inputs = {nn.Identity()()}
	local outputs = {}
	local gnode

	for t = 1, T do
		inputs[t+1] = nn.Identity()()

		if (t == 1) then
			gnode = hs[t]({inputs[t], inputs[t+1]})
		else
			gnode = hs[t]({inputs[t-1], inputs[t+1]})
		end
		outputs[t] = gs[t](gnode) -- gnode parent
	end
	return nn.gModule(inputs, outputs)
end

---------------[[ ETAPE 5 ]]---------------
-- Apprentissage par descente de gradient


function gradient_descent(model, v, nb_iter, lr, T, dim_x)

	local losses = nn.ParallelCriterion()
	for i=1, T do
		losses:add(nn.ClassNLLCriterion(), 1/T)
	end
	
	for it=1, nb_iter do
		print('ITERATION: ', it)

		for i = 1, v.nbatches do

			local inputs = {}
			inputs[1] = torch.zeros(dim_x)
			for k = 1, T do
				inputs[k+1] = char2vec(v, v.x_batches[i][{1,k}])
			end

			local labels = {}
			for k = 1, T do
				labels[k] = v.y_batches[i][{1,k}]
			end


			model:zeroGradParameters()
			out = model:forward(inputs)
			-- print('inputs:', decode_batch(v, v.x_batches[i]))
			-- print('labels:', decode_batch(v, v.y_batches[i]))
			-- print ('inputs: ', inputs)
			-- print('labels: ', labels)
			-- print('outputs:', decode_outputs(v, out))


			err = losses:forward(out, labels)
			delta = losses:backward(out, labels)
			model:backward(inputs, delta)
			model:updateParameters(lr)
		end
	end
	return model
end

T = 5
dim_h = 10 -- cross-val pour déterminer la taille de N ( taille de l'espace latent )

v, dim_x, nbatches = load_exemples(T)

print('dim_x', dim_x)
print('dim_h', dim_h)
print('T', T)




---------------[[ ETAPE 1 ]]---------------
g = create_g(dim_x, dim_h)
-- graph.dot(g.fg, 'module_g', 'module_g')

---------------[[ ETAPE 2 ]]---------------
h = create_h(dim_x, dim_h) 
-- graph.dot(h.fg, 'module_h', 'module_h')

---------------[[ ETAPE 3 ]]---------------
gs = model_utils.clone_many_times(g, T)
hs = model_utils.clone_many_times(h, T)

rnn = create_rnn(gs, hs, T)
-- graph.dot(rnn.fg, 'RNN', 'RNN')

-------[[Test des modules]]-----------

h0 = torch.Tensor(dim_h):fill(0)
h1 = h:forward({h0, char2vec(v, 1)})
-- print(h1)

g0 = create_g(dim_x, dim_h)
h2 = g:forward(h1)
-- print(h2)

model = create_rnn(gs, hs, T)
h3 = model:forward({h0, char2vec(v, 1), char2vec(v, 2), char2vec(v, 3), char2vec(v, 4), char2vec(v, 5)})
print (h3)

---------------[[ ETAPE 4 ]]---------------


local lr = 1e-2
nb_iter = 100

---------------[[ ETAPE 5 ]]---------------
rnn = gradient_descent(rnn, v, nb_iter, lr, T, dim_x)


-- torch.save('rnn', model)
-- rnn = torch.load('model')


---------------[[ ETAPE 6 ]]---------------
-- Utilisation / Inférence
-- logP -> P ( exp )
-- sampler -> torch.Multinomial

-- C = 10 -- init time
-- for t=1, C do
-- 	z=nn.gModule:forward({z,x[t]})
-- end

-- while (true)
-- 	lp = fmodule(forward(z))
-- 	probas = torch.exp(lp)
-- 	ch = torch.Multinomial(probas, 1)[1]
-- 	X = torch.Tensor(V):fill(0)
-- 	X[ch] = 1
-- 	z = gModule:forward({z, X})
-- end
-- print (data)


-- local  x_batches= {torch.zeros(dim_x)}
-- for i = 1, T do
-- 	x_batches[i+1] = data.x_batches[i]
-- end
-- print(x_batches)




-- Wx+b -> nn.Linear()
-- si W est diagonal -> 
import numpy as np
import random
import math
import copy


class NeuralNetwork:
    def __init__(self, layer_dims):
        '''
        :param layer_dims: list containing number of nodes in each layer, including input and output layer
        '''
        self.dim_input = layer_dims[0]
        self.dim_output = layer_dims[-1]
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.weights = self.initialize_weights()
        self.a_cache = {}
        self.z_final_cache = None
        self.d_w_cache = {}

    def initialize_weights(self):
        weights = {}
        #use He initialization for weights of hidden layers. (cos will use relu activation)
        for layer in range(1, len(self.layer_dims)-1):
            prev_layer_units = self.layer_dims[layer-1]
            curr_layer_units = self.layer_dims[layer]
            weights['w{}'.format(layer)] = np.random.randn(curr_layer_units, prev_layer_units) * math.sqrt(2.0/prev_layer_units)
        #initializate weights of output layer
        prev_layer_units = self.layer_dims[-2]
        curr_layer_units = self.layer_dims[-1]
        weights['w{}'.format(len(self.layer_dims)-1)] = np.random.randn(curr_layer_units, prev_layer_units) * math.sqrt(1.0/prev_layer_units)
        return weights

    def __str__(self):
        return str(self.weights)

    def get_num_layers(self):
        return self.num_layers

    def get_layer_dims(self):
        return self.layer_dims

    def fp(self, X, prob_random_select=0.1, choose_best_output=False):
        '''
        :param X: input, in shape (num_var, num_obs)
        :param prob_random_select: prob of selecting random output
        :param choose_best_output: if True, always select the node in the final layer with highest value - this overrides prob_random_select.
        :return: a_final, used in calculating advantage. y, representing predicted outcome/action
        '''
        N = self.num_layers
        a_prev = X
        self.store_a_in_cache(0, X) #for backprop
        for layer in range(1, N-1):
            z = self.fp_calc_z(layer, a_prev)
            a_curr = self.relu(z)
            self.store_a_in_cache(layer, a_curr)
            a_prev = a_curr
        #final layer
        z = self.fp_calc_z(N-1, a_prev)
        self.store_z_final_in_cache(z)
        a = self.softmax(z)
        a_final, y = self.fp_select_y(a, prob_random_select, choose_best_output)
        self.store_a_in_cache(N - 1, a_final)
        return a_final, y

    def store_z_final_in_cache(self, z):
        if self.z_final_cache is None:
            self.z_final_cache = z
        else:
            self.z_final_cache = np.hstack([self.z_final_cache, z])

    def store_a_in_cache(self, layer_num, a):
        a_layer_name = 'a' + str(layer_num)
        if a_layer_name in self.a_cache:
            self.a_cache[a_layer_name] = np.hstack([self.a_cache[a_layer_name], a])
        else:
            self.a_cache[a_layer_name] = a

    def fp_select_y(self, a, prob_random_select=0.1, choose_best_output=False):
        '''
        :param a: softmax activated units
        :param prob_random_select: prob that the output will be chosen at random, regardless of values in a
        :param choose_best_output: if True, always select the node with highest value
        :return: a_final. size (output_dim, m). Derived from a, but (output_dim - 1) cells have been set to 0.
        The non-0 cell represents the selected outcome, where selection probability is based on the probabilities in a.
        '''
        a = copy.deepcopy(a)
        num_obs = a.shape[1]
        y = np.zeros((1, num_obs))
        for obs_num in range(num_obs):
            obs_cumsum = np.cumsum(a[:, obs_num])
            random_num = random.uniform(0, obs_cumsum[-1]) #may not add up to 1, due to adding epsilon to denominator in softmax
            is_outcome_selected = False

            #randomly select action with probability prob_random_select
            if random.random()<prob_random_select:
                preselected_output = random.randint(0, len(obs_cumsum))
            else:
                preselected_output = -1

            #choose the best option
            if choose_best_output:
                preselected_output = a[:, obs_num].argmax()

            for outcome_num, prob in enumerate(obs_cumsum):
                if (random_num <= prob and not is_outcome_selected) or outcome_num == preselected_output:
                    is_outcome_selected = True
                    y[:,obs_num] = outcome_num
                else:
                    a[outcome_num, obs_num] = 0
        return a, y

    def fp_calc_z(self, layer_num, a_prev):
        weight = self.weights['w' + str(layer_num)]
        z = np.dot(weight, a_prev)
        return z

    def relu(self, z):
        a = copy.deepcopy(z)
        a[a < 0] = 0
        return a

    def softmax(self, z):
        a = np.exp(z)
        a_sum = np.sum(a, axis=0)
        a = a / (a_sum + 10**-10) #ideally, if a_sum==0, should set all probs for that obs to be the same
        return a

    def bp(self, advantage):
        #da and dw are needed for backprop to previous layers
        #only the most recent da, dz is needed, so only keep 1 da, dz at a time
        #dw is needed to update weights, so maintain a cache to store dw, under self.d_w_cache

        #final layer
        d_a_curr = self.bp_calc_d_a_final(advantage)
        d_z_curr = self.bp_calc_d_z_final(d_a_curr)
        d_w_curr = self.bp_calc_d_w(d_z_curr, self.a_cache['a' + str(self.num_layers-2)])
        self.update_d_w_cache(self.num_layers-1, d_w_curr)

        d_z_next = d_z_curr
        for layer in reversed(range(1, self.num_layers-1)):
            w_next = self.weights['w' + str(layer+1)]
            a_curr = self.a_cache['a' + str(layer)]
            a_prev = self.a_cache['a' + str(layer-1)]

            d_a_curr = self.bp_calc_d_a(w_next, d_z_next)
            d_z_curr = self.bp_calc_d_z_relu(d_a_curr, a_curr)
            d_w_curr = self.bp_calc_d_w(d_z_curr, a_prev)
            self.update_d_w_cache(layer, d_w_curr)

            d_z_next = copy.deepcopy(d_z_curr)

    def update_d_w_cache(self, layer_num, d_w_curr):
        d_w_layer_name = 'dw' + str(layer_num)
        self.d_w_cache[d_w_layer_name] = d_w_curr

    def bp_calc_d_w(self, d_z_curr, a_prev):
        return np.dot(d_z_curr, a_prev.T) / a_prev.shape[1]

    def bp_calc_d_a(self, w_next, d_z_next):
        return np.dot(w_next.T, d_z_next)

    def bp_calc_d_z_relu(self, d_a_curr, a_curr):
        return d_a_curr * (a_curr>0).astype('int')

    def bp_calc_d_z_final(self, d_a_final):
        z_final = self.z_final_cache
        a_final = self.a_cache['a' + str(self.num_layers-1)]
        d_z_final = np.zeros(d_a_final.shape)
        
        for obs_num in range(z_final.shape[1]):
            z_final_obs = copy.deepcopy(z_final[:,obs_num])
            z_final_obs_exp = np.exp(z_final_obs)
            a_final_obs = copy.deepcopy(a_final[:,obs_num])
            exp_of_selected_y = z_final_obs_exp[a_final_obs != 0][0]
            d_a_final_obs = self.find_non_zero_element(d_a_final[:,obs_num]) #d_z will be multiplied by the non-0 element of d_a_final_obs
            #for each node of z in each observation, there are 2 gradient formulas depending on whether that particular
            #node was eventually chosen as the prediction
            for node_num in range(len(z_final_obs_exp)):
                node_a_numerator = z_final_obs_exp[node_num]
                node_a_denom = np.sum(z_final_obs_exp)
                node_a_denom_sq = max(node_a_denom**2, 0.000000001) #in case node_a_denom = 0
                if a_final_obs[node_num] == 0:
                    d_z_curr_node = -(node_a_numerator*exp_of_selected_y) / node_a_denom_sq
                else:
                    d_z_curr_node = (node_a_denom*exp_of_selected_y - exp_of_selected_y**2) / node_a_denom_sq
                d_z_final[node_num, obs_num] = d_z_curr_node * d_a_final_obs
        return d_z_final

    def find_non_zero_element(self, arr):
        arr = arr[arr != 0]
        if len(arr) == 0:
            return 0
        else:
            return arr[0]

    def calc_loss(self, pi, advantage):
        return -np.log(pi) * advantage

    def bp_calc_d_a_final(self, advantage):
        d_a_final= copy.deepcopy(self.a_cache['a' + str(self.num_layers - 1)])
        d_a_final[np.abs(d_a_final)>0.0000001] = 1 / d_a_final[np.abs(d_a_final)>0.0000001]
        d_a_final *= -advantage
        return d_a_final

    def update_weights(self, learning_rate):
        #ideally should use other optimization techniques, but vanilla gradient descent will do here
        for layer in range(1, self.num_layers):
            self.weights['w{}'.format(layer)] -= (self.d_w_cache['dw{}'.format(layer)] * learning_rate)

    def clear_caches(self):
        self.a_cache = {}
        self.z_final_cache = None
        self.d_w_cache = {}


if __name__ == '__main__':
    nn = NeuralNetwork([2,3,4,3,2])
    a_final, y = nn.fp(np.array([[-1, -0.2, 0.2], [-3, -0.2, 0.2]]))
    a_final1, y1 = nn.fp(np.array([[-1, -0.2, 0.2], [-3, -0.2, 0.2]]))
    nn.bp(np.array([1, -1, 0.2, 1, -1, -0.2]))
    nn.update_weights(0.1)





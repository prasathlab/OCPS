import torch
import torch.nn as nn
import torch.nn.functional as F





class THR():
    def __init__(self, softmax, true_class, alpha):
        self.prob_output = softmax
        self.true_class = true_class
        self.alpha = alpha * (1 + (1/softmax.shape[0]))
        

    def conformal_score(self):
        conformal_score = self.prob_output[range(self.prob_output.shape[0]), self.true_class]
        return conformal_score
    

    def quantile(self):
        conformal_scores = self.conformal_score()
        quantile_value = torch.quantile(conformal_scores, self.alpha)
        return quantile_value


    def prediction(self, softmax, quantile_value):
        prob_output = softmax
        predictions = prob_output >= quantile_value
        predictions = predictions.int()
        return predictions
    












class RAPS():
    def __init__(self, softmax, true_class, alpha, k_reg, lambd, rand=True):
        self.prob_output = softmax
        self.true_class = true_class
        self.alpha = (1 - alpha) * (1 + (1 / softmax.shape[0]))
        self.k_reg = k_reg
        self.lambd = lambd
        self.rand = rand


    def conformal_score(self):
        conformal_score = []
        for i in range(self.prob_output.shape[0]):
            true_class_prob = self.prob_output[i][self.true_class[i]]
            current_class_prob = self.prob_output[i]
            sorted_class_prob, _ = torch.sort(current_class_prob, descending=True)
            index = torch.nonzero(sorted_class_prob == true_class_prob).item()
            cumulative_sum = torch.sum(sorted_class_prob[:index + 1])
            if index - self.k_reg > 0:
               cumulative_sum = cumulative_sum + self.lambd*(index - self.k_reg)
            
            if self.rand:
                U = torch.rand(1).item()
                cumulative_sum = cumulative_sum - U*sorted_class_prob[index]

            conformal_score.append(cumulative_sum)

        conformal_score = torch.tensor(conformal_score)

        return conformal_score



    def quantile(self):
        conformal_scores = self.conformal_score()
        quantile_value = torch.quantile(conformal_scores, self.alpha)
        return quantile_value
    


    def prediction(self, softmax, quantile_value):
        prob_output = softmax
        prediction = torch.zeros(prob_output.shape[0], prob_output.shape[1])
        for i in range(prob_output.shape[0]):
            current_class_prob = prob_output[i]
            sorted_class_prob, _ = torch.sort(current_class_prob, descending=True)
            sum = 0
            j = 0
            for idx in range(len(sorted_class_prob)):
                if sum <= quantile_value:
                    sum += sorted_class_prob[idx]
                    if idx - self.k_reg > 0:
                        sum = sum + self.lambd*(idx - self.k_reg)
                    j += 1
                else:
                    break
                
                if j != prob_output.shape[1]:
                    j += 1

            """
            
            if self.rand:
                U = torch.rand(1).item()
                if j != prob_output.shape[1]:
                    N = torch.sum(sorted_class_prob[:j + 1]) - quantile_value
                else:
                    N = torch.sum(sorted_class_prob[:j]) - quantile_value
                if idx - self.k_reg > 0:
                    N += self.lambd*(j - self.k_reg)

                if j != prob_output.shape[1]:
                    D = sorted_class_prob[j]
                else:
                    D = sorted_class_prob[j-1]
                if idx - self.k_reg > 0:
                    D += self.lambd

                if N/D <= U:
                    j = j -1
            """
            
            for idx in range(j):
                index = torch.nonzero(current_class_prob == sorted_class_prob[idx]).item()
                prediction[i][index] = 1.0
                
        return prediction






                







        




class APS():
    def __init__(self, softmax, true_class, alpha):
        self.prob_output = softmax
        self.true_class = true_class
        self.alpha = (1 - alpha) * (1 + (1 / softmax.shape[0]))
        
        

    def conformal_score(self):
        conformal_score = []
        for i in range(self.prob_output.shape[0]):
            true_class_prob = self.prob_output[i][self.true_class[i]]
            current_class_prob = self.prob_output[i]
            sorted_class_prob, _ = torch.sort(current_class_prob, descending=True)
            index = torch.nonzero(sorted_class_prob == true_class_prob).item()
            cumulative_sum = torch.sum(sorted_class_prob[:index + 1])
            conformal_score.append(cumulative_sum)
        conformal_score = torch.tensor(conformal_score)
        
        return conformal_score


    def quantile(self):
        conformal_scores = self.conformal_score()
        quantile_value = torch.quantile(conformal_scores, self.alpha)
        
        return quantile_value
        

    
    def prediction(self, softmax, quantile_value):
        prob_output = softmax
        prediction = torch.zeros(prob_output.shape[0], prob_output.shape[1])
        for i in range(prob_output.shape[0]):
            current_class_prob = prob_output[i]
            sorted_class_prob, _ = torch.sort(current_class_prob, descending=True)
            
            sum = 0
            
            for idx in range(len(sorted_class_prob)):
                if sum <= quantile_value:
                    sum += sorted_class_prob[idx]
                    index = torch.nonzero(current_class_prob == sorted_class_prob[idx]).item()
                    prediction[i][index] = 1.0
                else:
                    break
            

        return prediction
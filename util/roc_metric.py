from sklearn.metrics import roc_auc_score
import numpy as np

def compute_AUC(labels, scores):
    labels = labels.detach().cpu().numpy()
    labels = labels.astype(np.uint8)
    scores = scores.detach().cpu().numpy()
    
    return roc_auc_score(labels, scores)


        
    

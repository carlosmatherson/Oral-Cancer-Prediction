def mrmr(X, y, feature_names=None, k=10):

    n_features = X.shape[1]
    
    F, _ = f_classif(X, y)
    relevance = np.nan_to_num(F)
    
    selected = []
    selected_names = []
    remaining = list(range(n_features))
    
    best_idx = np.argmax(relevance)
    selected.append(best_idx)
    remaining.remove(best_idx)
    
    for i in range(1, k):
        best_score = -np.inf
        best_idx = -1
        
        for idx in remaining:

            correlations = []
            for sel_idx in selected:
                corr = np.abs(np.corrcoef(X[:, idx], X[:, sel_idx])[0, 1])
                correlations.append(corr)
            redundancy = np.mean(correlations) if correlations else 0
            
            score = relevance[idx] - redundancy
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        selected.append(best_idx)
        remaining.remove(best_idx)

        if feature_names is not None:
            selected_names = [feature_names[idx] for idx in selected]
    
    return selected_names

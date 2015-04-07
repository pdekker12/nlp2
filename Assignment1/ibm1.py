

def train_ibm1(parallel_sentences, n_iterations):
    #### Based on Collins, figure 2. Simplified from IBM 2 to IBM 1
    
    # TODO: Initialize t(f|e)
    
    # Perform n_iterations
    for s in range(0,n_iterations):
        # TODO: Set all counts c(...) = 0
        
        # For every sentence in the corpus
        for k in range(1,len(parallel_sentences)):
            pair = parallel_sentences[k]
            f = pair[0]
            e = pair[1]
            m_k = len(f)
            l_k = len(e)
            
            for i in range(1,m_k):
                for j in range(0,l_k):
                    
                    # TODO: Compute all c values
        
        # TODO: Set t(f|e) =  c(e,f)/c(e)
    
    # TODO: Return t(f|e)
            

# Dynamic-selftraining-GCN


*******************************************************************
This code is based on the GCN code in kipf's version , implemented mainly on tensorflow.
*******************************************************************
How to run the experiments?
    
    "cd" where "exps.py" is, and
    "python train_pipeline.py", 
    Wait for the results!
    
    Or run it on server and nohup it:
    "nohup python3 -u train_pipeline.py  > log.txt 2>&1 &"
    
    Results can be seen in "results.txt".

*******************************************************************
How to modify the experiments you want?

    See the "exps.py", 
    "results=main(inputParameters=parameters)" is a standard form to convey parameters into 
    experiment, "parameters" is a dictionary which restore value of parameters.
    
    Default parameters can be seen in function "constructDefaultParameterDict", if a parameter 
    is not in the keys of "parameters", we use its default value.
    
    For example:
    "for parameters in [{'dataset':'cora','train_size':20},
                        {'dataset':'pubmed','standard_split':True}]:
        
        for i in range(100):
            
            results=main(inputParameters=parameters)"

import argparse
import utils.extractive_models
parser = argparse.ArgumentParser(description = "Train an abstractive model on the EUR-Lex dataset which is pre-processed with an extractive model at a certain extractive compression ratio.")

parser.add_argument('extractive_model', type= str, 
                        help= "The extractive model to be used for pre-processing the dataset.")
parser.add_argument('compression_ratio', type= int, default= 4, choices= range(1, 10),
                        help= "The compression ratio to be used for the extractive model. Is in the form of an integer where 5 is 0.5, 9 is 0.9, etc.")
parser.add_argument('abstractive_model', type= str,
                        help= "The abstractive model to be used for fine-tuning.")
    
    #Optional arguments
parser.add_argument('-m', '--mode', choices= ['fixed', 'dependent', 'hybrid'], type= str, default= 'fixed',
                        help= "The ratio mode to use for the extractive summarization stage.")
parser.add_argument('-lr', '--learning_rate', type= float, default= 5e-5, metavar= "",
                        help= "The learning rate to train the abstractive model with.")
parser.add_argument('-e', '--epochs', type= int, default= 40, metavar= "",
                        help= "The amount of epochs to train the abstractive model for.")
parser.add_argument('-b', '--batch_size', type= int, default= 8, metavar= "",
                        help= "The batch size to train the abstractive model with.")
parser.add_argument('-w', '--warmup_ratio', type= float, default= 0.1, metavar= "",
                        help= "The warmup ratio to train the abstractive model for.")
parser.add_argument('-v', '--verbose', action= "store_false", default= True,
                        help= "Turn verbosity on or off.")
parser.add_argument('-wd', '--weight_decay', type= float, default= 0.01, metavar= "",
                        help= "The weight decay to train the abstractive model with.")
parser.add_argument('-lbm', '--load_best_model_at_end', action= "store_false", default= True,
                        help= "Load the best model at the end of training.")
parser.add_argument('-es', '--early_stopping_patience', type= int, default= 5, metavar= "",
                        help= "The amount of patience to use for early stopping.")
parser.add_argument('-mfm', '--metric_for_best_model', type= str, default= "eval_loss", metavar= "",
                        help= "The metric to use for selection of the best model.")
parser.add_argument('-p', '--peft', action= "store_true", default= False, 
                        help= "Use PEFT for training.")    
parser.add_argument('-ne', '--no_extraction', action= "store_true", default= False,
                        help= "Finetune a model on the whole dataset without any extractive steps.")                
    
args = parser.parse_args()  

extractive_model, extractive_tokenizer = utils.extractive_models.select_extractive_model(args.extractive_model)
#print all arguments
for arg in vars(args):
    print (arg, getattr(args, arg))
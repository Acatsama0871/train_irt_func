
from __future__ import absolute_import, division, print_function, unicode_literals

from data_batch_aug import CL_Data_loader
from data_batch import Data_loader
from irt_dataloader import CL_dff_loader
from py_irt import scoring

# use masking
from model_fragment_context import Text_Encoder, FastSR, Parser, Attention, Context_Attention
#from data_batch import Data_loader
import os
import glob
import time
import random
import pdb
import argparse
import pickle
import numpy as np
#from nltk import sent_tokenize, word_tokenize
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd
from matplotlib import pyplot as plt

import torch
import argparse
#from torchinfo import summary
#from  torch.cuda.amp import autocast

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# In[2]:

# implementing CL schedule
    
    

def estimate_theta(the_model, eval_loader, model_eval_loader, n_eposides_valid, device):
    
    # estimate difficult level       
    valid_pred = []
    valid_label = []
    valid_diffs = []
    
    all_items = []
    
    for k in range(n_eposides_valid):

        the_model.eval()

        # retrieve data from IRT loader
        
        # items is a dictionary with keys:
        #ID, Pos_support_locs	Neg_support_locs	Query_loc	Label	Alpha	Aug_type	diff
        items = model_eval_loader.next_batch()
        
        all_items.append(items)
        
        #print(items)
        cur_diff = np.array(items['diff'])
        #print(cur_diff)
        
        cur_x_support, cur_x_query, cur_y_query = eval_loader.next_batch_by_irt_items(items)
        
        cur_x_support = [item.to(device) for item in cur_x_support] 
        cur_x_query = [item.to(device) for item in cur_x_query]
        cur_y_query = cur_y_query.to(device)
            
        # evaluation
        with torch.no_grad():
            p_y, loss = the_model(cur_x_support, cur_x_query, cur_y_query)
        
        p_y = p_y.detach().cpu().numpy()
        y_hat = p_y.argmax(axis = -1)
    
        # metrics
        valid_pred.append(y_hat)
        valid_label.append(cur_y_query.detach().cpu().numpy())
        valid_diffs.append(cur_diff)

    # # estimate theta
    valid_diffs = np.concatenate(valid_diffs)
    valid_pred = np.concatenate(valid_pred)
    valid_label = np.concatenate(valid_label)
    
    #print(valid_diffs.shape, valid_pred.shape, valid_label.shape)
    
    cur_resp = np.where(valid_pred == valid_label, 1, 0)
    cur_theta = scoring.calculate_theta(difficulties=valid_diffs, response_pattern=cur_resp)[0]
    
    pickle.dump([all_items, valid_diffs, valid_pred, valid_label, cur_resp], open("test_set.pkl","wb"))
    print("dump saved!")

    return cur_theta
        
    
            
# train function - single class
def train_mnet(the_model, args, train_loader, valid_loader, \
               irt_train_loader, irt_eval_loader, the_device,\
              model_path = '.', model_name = 'model.pth', theta_subsample_size=100):

    print(the_model)
    print(the_model.encoder)
    if args.use_fragment:
        print(the_model.att_encoder)

    best_acc = 0
    #best_loss = np.Inf
    cnt = 0

    # move model
    the_model = the_model.to(the_device)

    #the_model = the_model.half().to(the_device)

    # history
    history = {'train_acc': [], 'train_loss': [],
             'train_acc_avg': [], 'train_loss_avg': [],
             'valid_acc': [], 'valid_loss': [],
             'valid_acc_avg': [], 'valid_loss_avg': [],
              'theta': []}

    # set up optimizer 
    optimizer = torch.optim.Adam(the_model.parameters(), lr=args.learning_rate)

    #optimizer = torch.optim.SGD(the_model.parameters(), lr=args.learning_rate)
    # if schedule learning rate
    if args.learning_rate_schedule is True:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=args.update_step,
                                                gamma=args.gamma)
    update_count = 0
    if args.verbose:
        print(f"Initial learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

    # Start training
    model_train_data = irt_train_loader.batch_generator_without_theta(subsample_size=theta_subsample_size)
    cur_theta = 0
    
    for i in range(args.epochs):

        cur_theta = estimate_theta(the_model, train_loader, model_train_data, 25, the_device)
        history['theta'].append(cur_theta)
        
        print("epoch: {0}, theta: {1}".format(i, cur_theta))
        
        cur_irt_data = irt_train_loader.batch_generator_with_theta(theta=cur_theta)            

        # train metrics
        train_cumulative_acc = 0
        train_cumulative_loss = 0

        the_model.train()

        for j in range(args.training_eposides):

            if args.CL:
                
                items = cur_irt_data.next_batch()
                #print(items)

                cur_x_support, cur_x_query, cur_y_query = train_loader.next_batch_by_irt_items(items) 

            else:
                cur_x_support, cur_x_query,cur_y_query = train_loader.next_batch()

            cur_x_support = [item.to(the_device) for item in cur_x_support] 
            cur_x_query = [item.to(the_device) for item in cur_x_query]
            cur_y_query = cur_y_query.to(the_device)

            p_y, cur_loss = the_model(cur_x_support, cur_x_query, cur_y_query)
        
            # backward
            cur_loss.backward()
            #torch.nn.utils.clip_grad_norm_(the_model.parameters(), 1)

            optimizer.step()
            optimizer.zero_grad()

            # learning rate schedule
            if args.learning_rate_schedule is True:
                scheduler.step()
                update_count += 1

                if update_count % args.update_step == 0 and args.verbose:
                    print(f"Update learning rate to {optimizer.state_dict()['param_groups'][0]['lr']}")

            # cumulative loss and acc
            p_y = p_y.detach().cpu().numpy()      
            y_hat = p_y.argmax(axis = -1)
            cur_acc = (cur_y_query.cpu().numpy().reshape(-1) == y_hat).astype(int).mean()

            train_cumulative_acc += cur_acc
            train_cumulative_loss += cur_loss.item()

            # record
            history['train_acc'].append(cur_acc)
            history['train_loss'].append(cur_loss.item())
        
        # update theta
        #cur_theta = estimate_theta(the_model, train_loader, model_eval_data, args.validation_eposides, the_device)
        #cur_theta = estimate_theta(the_model, train_loader, model_eval_data, 25, the_device)
        
        # evaluation metrics
        valid_cumulative_acc = 0
        valid_cumulative_loss = 0

        valid_pred = []
        valid_label = []

        for k in range(args.validation_eposides):
            the_model.eval()
            # retrieve data from validation loader
            data_batch = next(valid_loader)

            cur_x_support = [item.to(the_device) for item in data_batch[0]] 
            cur_x_query = [item.to(the_device) for item in data_batch[1]]
            cur_y_query = data_batch[2].to(the_device)

            # evaluation
            with torch.no_grad():
                p_y, loss = the_model(cur_x_support, cur_x_query, cur_y_query)

            p_y = p_y.detach().cpu().numpy()
            y_hat = p_y.argmax(axis = -1)

            # metrics
            valid_pred.append(y_hat)
            valid_label.append(data_batch[2].cpu().numpy())
            #valid_cumulative_acc += acc
            valid_cumulative_loss += loss.item()

        valid_pred = np.concatenate(valid_pred).reshape(-1)
        valid_label = np.concatenate(valid_label).reshape(-1)

        cur_acc = (valid_pred == valid_label).astype(int).sum()/len(valid_label)
        cur_loss = valid_cumulative_loss / args.validation_eposides

        # record
        history['train_acc_avg'].append(train_cumulative_acc / args.training_eposides)
        history['train_loss_avg'].append(train_cumulative_loss / args.training_eposides)
        history['valid_acc_avg'].append(cur_acc)
        history['valid_loss_avg'].append(valid_cumulative_loss / args.validation_eposides)

        # verbose
        if args.verbose:
            print('=' * 10 + f'Epoch: {i + 1} / {args.epochs}' + '=' * 10)
            print(f'\nTrain acc: {(train_cumulative_acc / args.training_eposides):.4f}, Train loss: {(train_cumulative_loss / args.training_eposides):.4f}')
            print(f'\nValidation acc: {cur_acc:.4f}, Validation loss: {(valid_cumulative_loss / args.validation_eposides):.4f}')

            print(classification_report(y_true=valid_label, y_pred=valid_pred))
            print('=' * 37)

        # early stopping
        
        #if cur_loss < best_loss:
        if cur_acc > best_acc:
            best_acc = cur_acc
            #best_loss = cur_loss
            torch.save(the_model.state_dict(), os.path.join(model_path, model_name))

            # calculate precision, recall, and f1
            p,r,f1,_ = precision_recall_fscore_support(y_true=valid_label, y_pred=valid_pred)
            history["prf"] = [p,r,f1]
            history["best_loss"] = cur_loss
            history["best_acc"] = cur_acc

            print("acc: {0:.4f}, loss: {1:.4f}, model saved!".format(cur_acc.item(), cur_loss))
            print("\n")

            cnt = 0
        else:
            cnt +=1
            if cnt==args.patience:
                print("stop training")
                print("Precision/recall: ", history["prf"])
                print("\n\n")
                break

    
    # print theta curve
    plt.plot(range(len(history['theta'])), history['theta'], "-")
    plt.title("Model theta")
    plt.show()
    
    return history           



def train_mnet_allClasses(args, train_loaders, valid_loaders, \
                          irt_train_loaders, irt_eval_loaders, \
                          the_device, model_path = '.', theta_subsample_size=100):


  # extract the names
  names = list(train_loaders.keys())

  # build models
  model_list = []
  for _ in range(len(names)):
    
    if args.use_fragment:
      att_encoder = Attention(emb_dim = args.emb_dim, seq_len = args.seq_len,  att_dim = args.att_dim)
    else:
      att_encoder = None

    if args.use_context:
      context_encoder = Context_Attention(emb_dim = args.context_dim, \
                                          att_dim = args.context_att_dim)
    else:
      context_encoder = None

    encoder = Text_Encoder(emb_dim=args.emb_dim, seq_len=args.seq_len, lstm_units=args.lstm_units, \
                              num_filters=args.num_filters, kernel_sizes=args.kernel_sizes, num_classes=args.n_way,\
                              use_syntax = args.use_syntax, syntax_dim = args.syntax_dim, \
                              use_context = args.use_context, context_dim = args.context_dim,\
                              syntax_num_filters = args.syntax_num_filters, context_output_dim = args.context_output_dim
                              )

    cur_model = FastSR(encoder, dist_metric=args.distance, 
               att_encoder = att_encoder, use_fragment = args.use_fragment, att_weight = args.att_weight,\
               use_syntax = args.use_syntax, 
               use_context = args.use_context, context_encoder = context_encoder, context_weight = args.context_weight,
               aux_weight = args.aux_weight)

    model_list.append(cur_model)

  # print model
  print(cur_model)

  if args.use_context:
    print(cur_model.context_encoder)
    #print(summary(cur_model.context_encoder,[(32, args.context_dim),\
    # (32, args.n_way*args.k_shot, args.context_dim)]))
  

  if args.use_fragment:
    print(cur_model.att_encoder)
    #print(summary(att_encoder,[(32, args.seq_len, args.emb_dim), \
    #  (32, 2,5, args.seq_len, args.emb_dim),  (32, args.seq_len),  (32,2,5, args.seq_len)]))



  # train models
  historys = {}
  for i in range(len(names)):
    # extract
    cur_name = names[i]
    cur_model = model_list[i]

    if args.save_model:    

      if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # train
    if args.verbose:
      print(f'\n\n{cur_name} model training start.\n')
    
    #print(type(irt_loaders[cur_name]))
    
    cur_history = train_mnet(cur_model, args = args,
                              train_loader=train_loaders[cur_name],
                              valid_loader=valid_loaders[cur_name],
                              irt_train_loader = irt_train_loaders[cur_name],
                              irt_eval_loader = irt_eval_loaders[cur_name],
                              the_device=the_device,
                              model_path= model_path,
                              model_name = cur_name + '.pth', theta_subsample_size=theta_subsample_size)
    historys[cur_name] = cur_history
    if args.verbose:
      print(f'\n{cur_name} model training finish.\n')
  
  # get average result
  #avg_accs = np.array([historys[i]['valid_acc_avg'] for i in historys]).mean(axis=0)
  #avg_loss = np.array([historys[i]['valid_loss_avg'] for i in historys]).mean(axis=0)

  return historys

def train(model_path, device,
          use_syntax = False, use_context = False, 
          aux_weight = 0.0, use_fragment = False,
          config = {}, 
          sample_path ='.',
          irt_path ='../data/IRT',
          train_data_path ='../data/train_bert_emb', theta_subsample_size=100):

  torch.autograd.set_detect_anomaly(True)

  args = Parser()
  
  args.set_option("use_syntax", use_syntax)
  args.set_option("use_context", use_context)
  args.set_option("aux_weight", aux_weight)
  args.set_option("use_fragment", use_fragment)

  # update model configure parameters
  for key in config:
    args.set_option(key, config[key])

  feature_path = train_data_path
  model_path = model_path

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  # only indexes of the sentences are kept in the dictionary
  train_pos_set = pickle.load(open(os.path.join(feature_path, "train_pos_dict.pkl"),"rb"))
  train_neg_set = pickle.load(open(os.path.join(feature_path, "train_neg_dict.pkl"),"rb"))
  val_pos_set = pickle.load(open(os.path.join(feature_path, "val_pos_dict.pkl"),"rb"))
  val_neg_set = pickle.load(open(os.path.join(feature_path, "val_neg_dict.pkl"),"rb"))

  data = []
  wordvectors = np.load(os.path.join(feature_path, "train_wordvectors.npy"))

  aug_wordvectors = np.load(os.path.join(feature_path, "wordvector_masked_aug.npy"))
  
  if args.sentence_mask:
    masks = np.load(os.path.join(feature_path, "sentence_masks.npy"))
    wordvectors = wordvectors * masks[:,:, None]
  else:    # assume emb also masked
    masks = np.all(wordvectors ==0, axis = -1).astype(int)
  
  data.append(wordvectors)

  emb_dim = wordvectors.shape[-1]
  seq_len = wordvectors.shape[-2]
  args.set_option("emb_dim", emb_dim)
  args.set_option("seq_len", seq_len)

  if use_syntax :
    syntax = np.load(os.path.join(feature_path, "train_syntax.npy"))
    data.append(syntax)
    syntax_dim = syntax.shape[-1]
    args.set_option("syntax_dim", syntax_dim)

    
  if use_context:
    section_prob = np.load(os.path.join(feature_path, "section_prob.npy"))
    data.append(section_prob)
    context_dim = section_prob.shape[-1]
    args.set_option("context_dim", context_dim)
                      
  columns = ["Perspective",
            "Study Period",
            "Country",
            'Sample Size',
            'Population',
            'Intervention',
      ]
  args.set_option("labels",  columns)
    
  # load IRT
  print(irt_path)
  #print(os.path.join(irt_path, "irt.pkl"))
  irt_train = pickle.load(open(os.path.join(irt_path, "irt_train.pkl"), 'rb'))
  irt_eval = pickle.load(open(os.path.join(irt_path, "irt_eval.pkl"), 'rb'))
    

  print("\n======== training argument ===========\n")
  args.print_options()

  # create data loader
  train_loaders = {}
  valid_loaders = {}
  irt_train_loaders = {}
  irt_eval_loaders = {}

  for cur_name in columns:
    print("use_fragment:", use_fragment)
    
    # fragment is class specific. need to change by class
    class_data = data.copy()
    print("len of general data: ", len(class_data))
    
    if use_fragment:
      frag_dict = pickle.load(open(os.path.join(feature_path, "frag_token_ids.pkl"), 'rb'))
      
      # mask for fragments in support set
      class_data.append(frag_dict[cur_name]) # num of train sample x sent_length
      
      # masks for query/supports
      class_data.append(masks)
    
    print("len of class specific data: ", len(class_data))

    # train
    
    cur_train_loader = CL_Data_loader(X_train_pos = train_pos_set[cur_name], X_train_neg = train_neg_set[cur_name], \
                                    X_val_pos = val_pos_set[cur_name], X_val_neg = val_neg_set[cur_name],\
                                    data = class_data, aug_data = aug_wordvectors, 
                                    batch_size = args.batch_size, k_shot = args.k_shot, train_mode= True)
    
    #cur_training_generator = cur_train_loader.next_batch_gen(return_sample_ids = False)

    # valid
    cur_eval_loader = Data_loader(X_train_pos = train_pos_set[cur_name], X_train_neg = train_neg_set[cur_name], \
                                    X_val_pos = val_pos_set[cur_name], X_val_neg = val_neg_set[cur_name],\
                                    data = class_data, batch_size = args.batch_size, k_shot = args.k_shot, train_mode = False)
    cur_validation_generator = cur_eval_loader.next_eval_batch_gen()
    
    # IRT
    cur_irt_train_loader = CL_dff_loader(irt_train[cur_name], args.batch_size)
    cur_irt_eval_loader = CL_dff_loader(irt_eval[cur_name], args.batch_size)

    # append result
    train_loaders[cur_name] = cur_train_loader
    valid_loaders[cur_name] = cur_validation_generator
    irt_train_loaders[cur_name] = cur_irt_train_loader
    irt_eval_loaders[cur_name] = cur_irt_eval_loader
    
    #print(type(cur_irt_loader))
    #print(cur_irt_loader.batch_generator_with_theta(0))

  print("\n======== start training ===========\n")

  torch.cuda.empty_cache()

  result_history = train_mnet_allClasses(args,
                                          train_loaders=train_loaders,
                                          valid_loaders=valid_loaders,
                                          irt_train_loaders = irt_train_loaders,
                                          irt_eval_loaders = irt_eval_loaders,
                                          the_device=device,
                                          model_path=model_path, theta_subsample_size=theta_subsample_size)
  
  print("\n========training result ===========\n")
  result = []
  for name in result_history:
    d = [name] + [item[1] for item in result_history[name]['prf']] + [result_history[name]['best_loss'], result_history[name]['best_acc']]
    result.append(d)

  result = pd.DataFrame(result, columns =["name","pre","rec","f1","loss","acc"])
  print(result)
  print(result['f1'].mean())
  print(result['loss'].mean())

  result.to_csv(os.path.join(model_path, "result.csv"))
  pickle.dump(result_history, open(os.path.join(model_path, "train_hist.pkl"),'wb'))
  

def scan_articles(model_path, result_path, device,
                  use_syntax = False, use_context = False,
                  aux_weight = 0.0, use_fragment = False,
                  batch =10,
                  config = {},
                  test_data_path ='../data',
                  test_emb = 'test_biobert_vectors',
                  train_data_path ='../data/train_bert_emb'
                  ):

  args = Parser()
  
  args.set_option("use_syntax", use_syntax)
  args.set_option("use_context", use_context)
  args.set_option("aux_weight", aux_weight)
  args.set_option("use_fragment", use_fragment)

  # update model configure parameters
  for key in config:
    args.set_option(key, config[key])

  feature_path = train_data_path
  model_path = model_path

  # only indexes of the sentences are kept in the dictionary
  train_pos_set = pickle.load(open(os.path.join(feature_path, "train_pos_dict.pkl"),"rb"))
  train_neg_set = pickle.load(open(os.path.join(feature_path, "train_neg_dict.pkl"),"rb"))
  val_pos_set = pickle.load(open(os.path.join(feature_path, "val_pos_dict.pkl"),"rb"))
  val_neg_set = pickle.load(open(os.path.join(feature_path, "val_neg_dict.pkl"),"rb"))

  data = []
  wordvectors = np.load(os.path.join(feature_path, "train_wordvectors.npy"))
  
  if args.sentence_mask:
    masks = np.load(os.path.join(feature_path, "sentence_masks.npy"))
    wordvectors = wordvectors * masks[:,:, None]
  else:
    masks = np.all(wordvectors ==0, axis = -1).astype(int)
  
  data.append(wordvectors)

  emb_dim = wordvectors.shape[-1]
  seq_len = wordvectors.shape[-2]
  args.set_option("emb_dim",  emb_dim)
  args.set_option("seq_len", seq_len)

  if use_syntax :
    syntax = np.load(os.path.join(feature_path, "train_syntax.npy"))
    data.append(syntax)
    syntax_dim = syntax.shape[-1]
    args.set_option("syntax_dim", syntax_dim)

    
  if use_context:
    section_prob = np.load(os.path.join(feature_path, "section_prob.npy"))
    data.append(section_prob)
    context_dim = section_prob.shape[-1]
    args.set_option("context_dim", context_dim)
                      
  columns = ["Perspective",
            "Study Period",
            "Country",
            'Sample Size',
            'Population',
            'Intervention',
      ]
  args.set_option("labels", columns)

  print("\n======== model argument ===========\n")
  args.print_options()

  # create data loader
  data_loaders = {}

  for cur_name in columns:

    class_data = data.copy()
    print("len of general data: ", len(class_data))
    
    if use_fragment:
      frag_dict = pickle.load(open(os.path.join(feature_path, "frag_token_ids.pkl"), 'rb'))
      
      # mask for fragments in support set
      class_data.append(frag_dict[cur_name]) # num of train sample x sent_length
      
      # masks for query/supports
      class_data.append(masks)
    
    print("len of class specific data: ", len(class_data))
        
    # valid
    cur_eval_loader = Data_loader(X_train_pos = train_pos_set[cur_name], X_train_neg = train_neg_set[cur_name], \
                                    X_val_pos = val_pos_set[cur_name], X_val_neg = val_neg_set[cur_name],\
                                    data = class_data, batch_size = args.batch_size, k_shot = args.k_shot, train_mode = False)
    #cur_validation_generator = cur_eval_loader.next_eval_batch_gen()

    # append result
    data_loaders[cur_name] = cur_eval_loader

  # load models

  models = {}
  for cur_label in columns:
    # construct model
    
    if args.use_fragment:
      att_encoder = Attention(emb_dim = args.emb_dim, seq_len = args.seq_len,  att_dim = args.att_dim)
    else:
      att_encoder = None

    if args.use_context:
      context_encoder = Context_Attention(emb_dim = args.context_dim, \
                                          att_dim = args.context_att_dim)
    else:
      context_encoder = None

    encoder = Text_Encoder(emb_dim=args.emb_dim, seq_len=args.seq_len, lstm_units=args.lstm_units, \
                              num_filters=args.num_filters, kernel_sizes=args.kernel_sizes, num_classes=args.n_way,\
                              use_syntax = args.use_syntax, syntax_dim = args.syntax_dim, \
                              use_context = args.use_context, context_dim = args.context_dim,\
                              syntax_num_filters = args.syntax_num_filters, context_output_dim = args.context_output_dim
                              )

    cur_model = FastSR(encoder, dist_metric=args.distance, 
               att_encoder = att_encoder, use_fragment = args.use_fragment, att_weight = args.att_weight,\
               use_syntax = args.use_syntax, 
               use_context = args.use_context, context_encoder = context_encoder, context_weight = args.context_weight,
               aux_weight = args.aux_weight)
    
    cur_model.load_state_dict(torch.load(os.path.join(model_path, cur_label + '.pth')))
    cur_model.eval()
    #cur_model.half().to(device)
    cur_model.to(device)

    models[cur_label] = cur_model

  # print model

  if args.use_context:
    print(cur_model.context_encoder)
    #print(summary(cur_model.context_encoder,[(32, args.context_dim),\
    # (32, args.n_way*args.k_shot, args.context_dim)]))
  

  if args.use_fragment:
    print(cur_model.att_encoder)
    #print(summary(att_encoder,[(32, args.seq_len, args.emb_dim), \
    #  (32, 2,5, args.seq_len, args.emb_dim),  (32, args.seq_len),  (32,2,5, args.seq_len)]))
    
  print(cur_model)  


  #if args.use_syntax or args.use_context:
  #  print(summary(encoder,[(32, args.seq_len, args.emb_dim),\
  #    (32, args.seq_len, args.syntax_dim), (32, args.context_dim)]))
  #else:
  #  print(summary(encoder,(32, args.seq_len, args.emb_dim)))


  # load test files
  
  if not os.path.exists(result_path):
    os.makedirs(result_path)
          
  tested_files = [x.split('.')[0] for x in glob.glob(result_path + '/*.csv')]
  test_files = pd.read_csv(os.path.join(test_data_path, "test_file_list.csv"))
  test_files = test_files["file_id"].astype(str).values.tolist()

  print("Total test files: ", len(test_files))

  processed_files = [os.path.basename(f).replace(".csv",'') for f in glob.glob(os.path.join(result_path, "*.csv"))]
  print("processed files: ", len(processed_files))

  # start scanning
  print("\n==========Start scanning============\n")

  cnt = 0 
  for filename in glob.glob(os.path.join(test_data_path, test_emb, '*.pkl')):

    cnt +=1 
    
    article_name = filename[(filename.rfind("/")+1):(filename.rfind("."))]

    if article_name in processed_files:
      continue

    start = time.time()

    test_df = pd.read_csv(os.path.join(test_data_path, 'test_csv', article_name+'.csv'))
    test_df = test_df.iloc[:, 0:2]
    test_df.columns = ["sid","sent"]

    if (article_name in test_files) and (article_name not in tested_files):
    # if (article_name in test_files):
      # test data prepare
      print(f'{cnt}: {article_name} start')
      #test_df = pd.read_csv(filename)
      pred_samples = pickle.load(open(filename, 'rb'))

      pred_data_set = []
      vectors = pred_samples["wordvector"]

      if args.sentence_mask:
        mask = pred_samples["mask"]
        #print("mask shape: ", mask.shape)
        vectors = vectors * mask[:,:, None]

      pred_data_set.append(vectors)

      print("word vector shape: ", pred_samples["wordvector"].shape)
            
      if args.use_syntax :
          syntax = pred_samples["syntax"]
          pred_data_set.append(syntax)
          #print("syntax shape: ", pred_samples["syntax"].shape)
        
      if args.use_context:
          context = pred_samples["section_prob"]
          pred_data_set.append(context)
          #print("section prob shape: ", pred_samples["section_prob"].shape)
      
      # for query, fragment/mask is all the words
      if args.use_fragment:
          # for biomed emb, there is no mask
          if "mask" in pred_samples:
              pred_data_set.append(pred_samples["mask"])
              pred_data_set.append(pred_samples["mask"])
          else:
              pred_sample_mask = np.all(pred_samples["wordvector"] == 0, axis = -1).astype(int)
              pred_data_set.append(pred_sample_mask)
              pred_data_set.append(pred_sample_mask)  

      atts = {}

      #print("number of features: ", len(pred_data_set))
      # zip to have all features for one sample
      pred_data = list(zip(*pred_data_set))
      
      for cur_label in columns:
        print(cur_label)
        # fetch the model
        the_model = models[cur_label]
        the_model.eval()
        # featch the data loader
        the_dataloader = data_loaders[cur_label]
        # result list
        cur_result = []

        if args.use_fragment:
          cur_atts = []

        #batch = 10     # predict by batch to speed up

        support_batch = [[] for i in range(len(pred_data_set))]
        query_batch = [[] for i in range(len(pred_data_set))]

        for i, v in enumerate(pred_data):

          # get testing support and query
          x_set, _, x_hat = the_dataloader.get_pred_set(v)

          for j in range(0, len(pred_data_set)):
            support_batch[j].append(x_set[j])
            query_batch[j].append(x_hat[j])

          if ((i+1)%batch == 0) or (i == len(vectors)-1):

            for j in range(0, len(pred_data_set)):
              support_batch[j] = np.concatenate(support_batch[j], axis = 0)
              query_batch[j] = np.concatenate(query_batch[j], axis = 0)
              #print("batch size: ", support_batch[j].shape, query_batch[j].shape)

              support_batch[j] = torch.Tensor(support_batch[j]).to(device)
              query_batch[j] = torch.Tensor(query_batch[j]).to(device)
            
              #support_batch[j] = torch.Tensor(support_batch[j]).half().to(device)
              #query_batch[j] = torch.Tensor(query_batch[j]).half().to(device)
            
            cur_y_query = torch.Tensor(np.zeros(query_batch[0].size(0)))  # faked target for loss computation
            cur_y_query = cur_y_query.to(device)
            #cur_y_query = torch.Tensor(np.zeros(query_batch[0].size(0)))  # faked target for loss computation
            #cur_y_query = cur_y_query.half().to(device)

            # forward
            with torch.no_grad():
              
              if args.use_fragment:
                cur_pred_prob, _, _, query_att = the_model(support_batch, query_batch, cur_y_query, return_att = True)
                query_att = query_att.reshape(-1, args.batch_size, *query_att.shape[1:])
                query_att = query_att.mean(axis = 1)
                cur_atts.append(query_att)

              else:
                cur_pred_prob, _ = the_model(support_batch, query_batch, cur_y_query)

            # append result
            cur_pred_prob = cur_pred_prob.cpu().numpy()
            cur_pred_prob = cur_pred_prob.reshape(-1, args.batch_size, 2)
            cur_result.append(cur_pred_prob[:,:,1].mean(axis =-1))
            
            # reset the list
            support_batch = [[] for i in range(len(pred_data_set))]
            query_batch = [[] for i in range(len(pred_data_set))]

        cur_result = np.concatenate(cur_result, axis = 0)
        #print("prediction shape: ", cur_result.shape)
        # add result to test df
        test_df[cur_label] = cur_result

        if args.use_fragment:
          cur_atts = np.concatenate(cur_atts, axis = 0)
          #print(cur_atts.shape)
          atts[cur_label] = cur_atts
    
      # save current article result
      test_df.to_csv(os.path.join(result_path, article_name + '.csv'), index=False)

      if args.use_fragment:
        pickle.dump(atts, open(os.path.join(result_path, article_name + '_att.pkl'), 'wb'))

      print(f'{article_name} finish in {time.time()-start} s\n')

  print("complete!")   


def generate_CL_samples(model_path, result_path, device,
                  use_syntax = False, use_context = False,
                  aux_weight = 0.0, use_fragment = False,
                  batch =10,
                  config = {},
                  test_data_path ='../data',
                  train_data_path ='../data/train_bert_emb'
                  ):

  args = Parser()
  
  args.set_option("use_syntax", use_syntax)
  args.set_option("use_context", use_context)
  args.set_option("aux_weight", aux_weight)
  args.set_option("use_fragment", use_fragment)

  # update model configure parameters
  for key in config:
    args.set_option(key, config[key])

  feature_path = train_data_path
  model_path = model_path

  # only indexes of the sentences are kept in the dictionary
  train_pos_set = pickle.load(open(os.path.join(feature_path, "train_pos_dict.pkl"),"rb"))
  train_neg_set = pickle.load(open(os.path.join(feature_path, "train_neg_dict.pkl"),"rb"))
  val_pos_set = pickle.load(open(os.path.join(feature_path, "val_pos_dict.pkl"),"rb"))
  val_neg_set = pickle.load(open(os.path.join(feature_path, "val_neg_dict.pkl"),"rb"))

  data = []
  wordvectors = np.load(os.path.join(feature_path, "train_wordvectors.npy"))
  
  if args.sentence_mask:
    masks = np.load(os.path.join(feature_path, "sentence_masks.npy"))
    wordvectors = wordvectors * masks[:,:, None]
  
  data.append(wordvectors)

  emb_dim = wordvectors.shape[-1]
  seq_len = wordvectors.shape[-2]
  args.set_option("emb_dim",  emb_dim)
  args.set_option("seq_len", seq_len)

  if use_syntax :
    syntax = np.load(os.path.join(feature_path, "train_syntax.npy"))
    data.append(syntax)
    syntax_dim = syntax.shape[-1]
    args.set_option("syntax_dim", syntax_dim)

    
  if use_context:
    section_prob = np.load(os.path.join(feature_path, "section_prob.npy"))
    data.append(section_prob)
    context_dim = section_prob.shape[-1]
    args.set_option("context_dim", context_dim)
                      
  columns = ["Perspective",
            "Study Period",
            "Country",
            'Sample Size',
            'Population',
            'Intervention',
      ]
  args.set_option("labels", columns)

  print("\n======== model argument ===========\n")
  args.print_options()

  # create data loader
  data_loaders = {}

  for cur_name in columns:

    if args.use_fragment:
      frag_dict = pickle.load(open(os.path.join(feature_path, "frag_token_ids.pkl"), 'rb'))
      data.append(frag_dict[cur_name]) # num of train sample x sent_length
      data.append(masks)

    # valid
    cur_loader = Data_loader(X_train_pos = train_pos_set[cur_name], X_train_neg = train_neg_set[cur_name], \
                                    X_val_pos = val_pos_set[cur_name], X_val_neg = val_neg_set[cur_name],\
                                    data = data, batch_size = args.batch_size, k_shot = args.k_shot, train_mode = True)
    #cur_validation_generator = cur_eval_loader.next_eval_batch_gen()

    # append result
    cur_loader_generator = cur_loader.next_batch_gen(return_sample_ids = True)
    data_loaders[cur_name] = cur_loader_generator

  # load models

  models = {}
  for cur_label in columns:
    # construct model
    
    if args.use_fragment:
      att_encoder = Attention(emb_dim = args.emb_dim, seq_len = args.seq_len,  att_dim = args.att_dim)
    else:
      att_encoder = None

    if args.use_context:
      context_encoder = Context_Attention(emb_dim = args.context_dim, \
                                          att_dim = args.context_att_dim)
    else:
      context_encoder = None

    encoder = Text_Encoder(emb_dim=args.emb_dim, seq_len=args.seq_len, lstm_units=args.lstm_units, \
                              num_filters=args.num_filters, kernel_sizes=args.kernel_sizes, num_classes=args.n_way,\
                              use_syntax = args.use_syntax, syntax_dim = args.syntax_dim, \
                              use_context = args.use_context, context_dim = args.context_dim,\
                              syntax_num_filters = args.syntax_num_filters, context_output_dim = args.context_output_dim
                              )

    cur_model = FastSR(encoder, dist_metric=args.distance, 
               att_encoder = att_encoder, use_fragment = args.use_fragment, att_weight = args.att_weight,\
               use_syntax = args.use_syntax, 
               use_context = args.use_context, context_encoder = context_encoder, context_weight = args.context_weight,
               aux_weight = args.aux_weight)
    
    cur_model.load_state_dict(torch.load(os.path.join(model_path, cur_label + '.pth')))
    cur_model.eval()
    #cur_model.half().to(device)
    cur_model.to(device)

    models[cur_label] = cur_model

  # print model

  if args.use_context:
    print(cur_model.context_encoder)
    #print(summary(cur_model.context_encoder,[(32, args.context_dim),\
    # (32, args.n_way*args.k_shot, args.context_dim)]))
  

  if args.use_fragment:
    print(cur_model.att_encoder)
    #print(summary(att_encoder,[(32, args.seq_len, args.emb_dim), \
    #  (32, 2,5, args.seq_len, args.emb_dim),  (32, args.seq_len),  (32,2,5, args.seq_len)]))
    
  print(cur_model)  


  #if args.use_syntax or args.use_context:
  #  print(summary(encoder,[(32, args.seq_len, args.emb_dim),\
  #    (32, args.seq_len, args.syntax_dim), (32, args.context_dim)]))
  #else:
  #  print(summary(encoder,(32, args.seq_len, args.emb_dim)))


  # load test files
  
  if not os.path.exists(result_path):
    os.makedirs(result_path)
          
  for cur_label in columns:
      
      the_model = models[cur_label]
      the_model.eval()
      # featch the data loader
      the_dataloader =data_loaders[cur_label]
        
      samples = []
        
      # start scanning
      print("\n==========Start generating 10K batches of {} sample pairs ============\n".format(cur_label))
 
      for i in range(batch):       
        
        cur_x_support, cur_x_query,cur_y_query, x_set_ids, x_hat_ids = next(the_dataloader)
        
        cur_x_support = [item.to(device) for item in cur_x_support] 
        cur_x_query = [item.to(device) for item in cur_x_query]
        cur_y_query = cur_y_query.to(device)
        with torch.no_grad():
            p_y, cur_loss = the_model(cur_x_support, cur_x_query, cur_y_query)
            p_y = p_y.detach().cpu().numpy()      
            
        samples.append([x_set_ids, x_hat_ids, cur_y_query.cpu().numpy(), p_y])
        
        if i%100 == 0:
            print("processed: ", i)
      
      pickle.dump(samples, open(os.path.join(result_path, cur_label +'_samples.pkl'), 'wb'))



  print("complete!")   



if __name__ == '__main__':
    
    
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    # to run use: python train_all_features.py --model_path=syntax --use_syntax=1 --use_section_prob=0 --use_other_feature=0
    ap.add_argument("-m", "--model_path", type=str, required=True, help="model path")
    ap.add_argument("-s", "--use_syntax", type=int, default=0, help="whether to use syntax")
    ap.add_argument("-p", "--use_context", type=int, default=0, help="whether to use section prob")
    ap.add_argument("-a", "--aux_weight", type=str, default="0.0", help="weight of aux target")
    ap.add_argument("-d", "--data_path", type=str, default='../data/', help="path to training data")
    ap.add_argument("-f", "--filters", type=int, default=20, help="number of filters")
    ap.add_argument("-l", "--lstm_units", type=int, default=100, help="LSTM memory units")
    ap.add_argument("-t", "--distance", type=str, default='normalized_cosine', help="distance measure")
    
    pars = vars(ap.parse_args())


    
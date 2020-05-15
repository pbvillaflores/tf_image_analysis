import os
import os.path
import shutil
import tensorflow as tf
import datareadinit
import numpy as np
import math 
import pprint 
import time 
LOGDIR = r'..\log_dir'   # <--- tensor board logs
DROP_RATE = 0.60
from datareadinit import train_x, train_y, test_x, test_y, train
from datareadinit import input_num_units, output_num_units, MAX_W, MAX_H
seed = 260

# Final conv layer output dimensions:
CONV_2 = ((MAX_W+1)//2 + 1)//2  # 47. The 2rd dim of the tensor output shape of conv
CONV_3 = ((MAX_H+1)//2 + 1)//2  # 36. The 3rd dim of the tensor output shape of conv
CONV_4 = 64                     #     The 4th dim of the tensor output shape of conv

start = time.time()
test_x2 = test_x.reshape(-1,MAX_W,MAX_H,1)
test_y2 = test_y.reshape(-1,1)
# x:(53, 188, 141, 1)  y: 53,1
test_res = []


# Effect of conv2d on input tensor shape:
# The conv2d has four hyperparameters:
#   Number of filters K
#   Spatial extent of filters F
#   The stride with which the filter moves S
#   The amount of zero padding P = (F-1) / 2 
#
# Produces a volume of size W2 x H2 x D2 where:
#   W2 = (W1 - F + 2*P)/S + 1
#   H2 = (H1 - F + 2*P)/S + 1
#   D2 = K


def conv_layer(input, size_in, size_out, name="conv"):
  with tf.name_scope(name):
  
    #                                    |--- F        |--- k
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1,seed=seed), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    with tf.device('/cpu:0'):
      #input=tf.Print(input, [input.shape],'input.shape' )
      
      #                                      |--- S
      conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
      #conv=tf.Print(conv,[ w.shape ], 'w.shape' )
      
    # act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    
    # Effect of maxpool on shape is (n+1)//2 on the 2 middle dimensions
    
    return tf.nn.max_pool(conv+b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def convb_layer(input, size_in, size_out, name="convb"):
  with tf.name_scope(name):
  
    #                                    |--- F        |--- k
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1,seed=seed), name="WB")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="BB")
    with tf.device('/cpu:0'):
      #input=tf.Print(input, [input.shape],'input.shape' )
      
      #                                      |--- S
      conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
      #conv=tf.Print(conv,[ w.shape ], 'w.shape' )
      
    tf.summary.histogram("weightsb", w)
    tf.summary.histogram("biasesb", b)
    
    return conv+b
	


    
def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1,seed=seed), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.matmul(input, w) + b
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act

def gen_model(learning_rate, use_two_fc, use_convs, hparam):
  global test_res
  
  tf.reset_default_graph()
  sess = tf.Session( )  # config=tf.ConfigProto(log_device_placement=True) )  to print device used

  # Setup placeholders, and reshape the data
  x = tf.placeholder(tf.float32, shape=[None, MAX_W, MAX_H, 1], name="x")
  # print ('MAX_W,MAX_H:',MAX_W,MAX_H)
  # MAX_W,MAX_H: 188 141
  # x_image = tf.reshape(x, [-1, MAX_W, MAX_H, 1])
  tf.summary.image('input', x, 10)  # parameter controls max number of impages
  y = tf.placeholder(tf.float32, shape=[None, output_num_units], name="labels")

  if use_convs==1:
    conv1 = conv_layer(x, 1, CONV_4, "conv")
    conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # expect result shape: ? x ((MAX_W+1)//2 + 1)//2 x ((MAX_H+1)//2 + 1)//2 x CONV_4 
  elif use_convs==2:
    conv1    = conv_layer(x, 1, 32, "conv1")
    # expect result shape: ? x (MAX_W+1)//2 x (MAX_H+1)//2 x 32 
    conv_out = conv_layer(conv1, 32, CONV_4, "conv2")
    # expect result shape: ? x ((MAX_W+1)//2 + 1)//2 x ((MAX_H+1)//2 + 1)//2 x CONV_4 
  else:
    convs = list ( range(use_convs-1) )
    convs[0] = x
    for i in range(use_convs-2):
      convs[i+1]=convb_layer(convs[i], 1, 1, "conv%d"%(i+3) )
  
  
    conv1    = conv_layer( convs[use_convs-2] , 1, 32, "conv1")
    conv_out = conv_layer(conv1, 32, CONV_4, "conv2")
  

  #flattened = tf.reshape(conv_out, [-1, CONV_2 * CONV_2 * CONV_4])
  #flattened = tf.reshape(conv_out, [-1,  int( (MAX_W*MAX_H)/CONV_2/CONV_2*1536 )  ])
  #flattened = tf.reshape(conv_out, [-1, CONV_2 * CONV_4])
  #drop_layer = tf.nn.dropout(conv_out, DROP_RATE, seed=seed )
  flattened = tf.reshape(conv_out, [-1, CONV_2 * CONV_4] )
  embedding_input = flattened
  embedding_size0 = train_x.shape[0] * ((MAX_H+1)//2 + 1)//2     # 16920
  embedding_size  = CONV_4 * CONV_2                              # 3008

  if use_two_fc:
    fc1 = fc_layer(flattened, CONV_2 * CONV_4, 1024, "fc1")
    #relu = tf.nn.relu(fc1)
    tf.summary.histogram("fc1", fc1)
    
    logits = fc_layer(fc1, 1024, 1, "fc2")
  else:
    
    logits = fc_layer(flattened, CONV_2*CONV_4, 1, "fc")

  with tf.name_scope('conv9'):
    weights = {
        'conv9': tf.Variable(tf.random_normal([ 36, output_num_units], seed=seed),name="w_c9")
    }
    # the first dimesion is probably is probably left over factor after dividing into the original dimensions
    
    biases = {
        'conv9': tf.Variable(tf.random_normal([output_num_units], seed=seed) ,name="b_c9")
    }
    # logits shape [192,1]
    #logit_reshape = tf.reshape(logits, [-1, CONV_4*3] )
    logit_reshape = tf.reshape(logits, [-1, 36] )
    #logit_reshape = tf.reshape(logits, [-1, 4*3] )
    

    conv9 = tf.matmul(logit_reshape,  weights['conv9'] ) + biases['conv9']
    
    output_layer = conv9 * .01231 / 5 + 1
    
    tf.summary.histogram("output_layer2", output_layer)
    #output_layer = tf.Print(output_layer, [conv9], 'conv9' )
  
  with tf.name_scope("mse"):
    #mse = tf.reduce_mean(
    #    tf.nn.softmax_cross_entropy_with_logits(
    #        logits=logits, labels=y), name="xent")
    
    
    #output_layer = tf.Print(output_layer, [output_layer ], 'output_layer' )
    #output_layer = tf.Print(output_layer, [y            ], 'y' )
    
    mse = tf.reduce_mean(tf.squared_difference( output_layer , y) )
    
    # [192,1] vs. [2,1]
    
    tf.summary.scalar("mse", mse)

  with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse)

  with tf.name_scope("accuracy"):
    #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32) )
    
    #accuracy = tf.reduce_mean(tf.squared_difference( logits , y) )
    accuracy = math.e ** (- mse**2 )
    
    tf.summary.scalar("accuracy", accuracy)

  summ = tf.summary.merge_all()


  #embedding = tf.Variable(tf.zeros([60, embedding_size]), name="test_embedding")  
  embedding = tf.Variable(tf.zeros([ embedding_size0, embedding_size]), name="test_embedding")
  
  assignment = embedding.assign(embedding_input)
  saver = tf.train.Saver()

  sess.run(tf.global_variables_initializer())
  #writer = tf.summary.FileWriter(LOGDIR + hparam)
  writer = tf.summary.FileWriter(LOGDIR + "\\" + hparam)
  writer.add_graph(sess.graph)
  #print(tf.get_default_graph().as_graph_def())   #<-- print our graph. result is very long
  
  config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
  embedding_config = config.embeddings.add()
  embedding_config.tensor_name = embedding.name
  embedding_config.metadata_path = datareadinit.LABELS
  tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
  '''
  config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
  embedding_config = config.embeddings.add()
  embedding_config.tensor_name = embedding.name
  embedding_config.sprite.image_path = SPRITES
  embedding_config.metadata_path = LABELS
  # Specify the width and height of a single thumbnail.
  embedding_config.sprite.single_image_dim.extend([MAX_W, MAX_H])
  tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
  '''
  max_epochs = 500
 
  total_batch = int(train_x.shape[0])
  print('total_batch', total_batch)
  for i in range(total_batch) :
    #print('train_x', train_x.shape, train_x)  #train_x (10, 376, 564) 
    #print('train_y', train_y.shape, train_y)  #train_y (10,) [0 0 0 0 1 1 2 0 0 0]
        
    batch_x, batch_y = train_x[i], train_y[i]
    
    # batch_x shape (2, 3393024)
    # batch_y shape (2, 1)
    
    if i % 20 == 0 or i==total_batch-1:
      [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch_x.reshape(-1, MAX_W,MAX_H,1), y: batch_y.reshape(-1,1) })
      print(i, "accuracy", train_accuracy )
      writer.add_summary(s, i)
    if i % 100 == 0 or i==total_batch-1:
      sess.run(assignment, feed_dict={x: train_x.reshape(-1, MAX_W,MAX_H,1) , y: train_y.reshape(-1,1) })
      saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
      # for the embedding.
    #print ('batch_x',batch_x.shape,batch_x)
    #print ('batch_y',batch_y.shape,batch_y)
    feed_dict={x: batch_x, y: batch_y}
    c = sess.run(train_step, feed_dict={x: batch_x.reshape(-1,MAX_W,MAX_H,1), y: batch_y.reshape(-1,1)})

  [mse_eval,s,output_eval] = sess.run([mse,summ,output_layer], feed_dict={x: test_x2, y: test_y2 })
  accuracy = math.e ** (- mse_eval**2 )
  print("test accuracy:",accuracy,"mse:",mse_eval)
  tf.summary.scalar("test_accuracy", accuracy)
  tf.summary.scalar("test_mse", mse_eval)
  
  print('Test data vs predictions')
  for i in range( 20 ):  # the first 20
  #for i in range( test_y2.shape[0] ):
    print('  ',i, datareadinit.test_fn[i], test_y2[i], output_eval[i]  )
    #print('  ',i, test_y2[i], output_eval[i]  )
    
  print('\n')
  test_res.append( [hparam, accuracy, mse_eval]  )
  

def make_hparam_string(learning_rate, use_two_fc, use_convs):
  conv_param = "conv=%d" % use_convs
  fc_param = "fc=2" if use_two_fc else "fc=1"
  return "lr_%.4E,%s,%s" % (learning_rate, conv_param, fc_param)

def main():
  # You can try adding some more learning rates
  for learning_rate in [1.35e-03]:
  #for ii in range(400):
  #  if ii < 200:
  #    learning_rate = 1e-02 + 5e-04*ii 
  #  else :
  #    learning_rate = 1e-04 + 5e-06*(ii-200)
    

    # Include "False" as a value to try different model architectures
    for use_two_fc in [False]:
      #for use_convs in [4,5]:
      for use_convs in [4]:
        # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2")
        hparam = make_hparam_string(learning_rate, use_two_fc, use_convs)
        print('Starting run for %s' % hparam)

        # Actually run with the new settings
        gen_model(learning_rate, use_two_fc, use_convs, hparam)
  print('Done training!')
  print('Test top results:')
  sort1 = sorted(test_res, key=lambda x:x[2] )
  pprint.pprint( sort1[:2] )
  
  print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

if __name__ == '__main__':
  main()
  print ('Runtime: %0.2f sec'% (time.time() - start))  
  
# http://projector.tensorflow.org/
# e\lr_1E-02,conv=2,fc=2 - nice distribution
# good acc:
#   e\lr_1E-02,conv=1,fc=2
#   e\lr_1E-02,conv=2,fc=1
#   e\lr_1E-04,conv=2,fc=1  0.9

#  lr_1E-04,conv=1,fc=1   0.521029637515 mse: 0.807433
#  lr_1E-03,conv=1,fc=1 0.187726998245 mse: 1.29335
#  lr_1E-02,conv=1,fc=2 0.462169151289 mse: 0.878535
# e\lr_1E-03,conv=1,fc=1

#
#Test top results:
#[['lr_1.3E-02,conv=1,fc=2', 0.63015850355744973, 0.67954683]
# ['lr_1.1E-02,conv=1,fc=2', 0.13639177651054477, 1.4114616],
# ['lr_1.5E-02,conv=1,fc=1', 0.10230156273448991, 1.5099107],
# ['lr_5.0E-03,conv=2,fc=1', 0.09948208885291436, 1.5191371],
# ['lr_1.1E-02,conv=1,fc=1', 0.04415906471134011, 1.76634]]

#
#lr_1.10E-02,conv=1,fc=2', 0.55446530447402231, 0.7679525
#lr_1.50E-02,conv=1,fc=2', 0.35884585933643431, 1.0123549
#lr_1.40E-02,conv=1,fc=2', 0.21564232792793195, 1.2386017
#lr_1.40E-02,conv=1,fc=1', 0.13625211543892238, 1.4118245


#
#[['lr_1.45E-02,conv=1,fc=2', 0.80816513265318157, 0.46150717]  <--
# ['lr_1.25E-02,conv=1,fc=2', 0.65508596037017042, 0.6503759],
# ['lr_1.15E-02,conv=1,fc=2', 0.63094118287272005, 0.67863292]
# ['lr_1.05E-02,conv=1,fc=2', 0.44181037741783924, 0.9038111],
# ['lr_1.45E-02,conv=1,fc=1', 0.14056536833897099, 1.4007436]]

#
#['lr_1.43E-02,conv=1,fc=2', 0.72784986283028874, 0.56361377],
#['lr_1.47E-02,conv=1,fc=2', 0.68196694631276045, 0.61868739],
#['lr_1.49E-02,conv=1,fc=2', 0.63598786492525383, 0.67273754],
#['lr_1.41E-02,conv=1,fc=1', 0.24297065177367114, 1.1894598],
#['lr_1.44E-02,conv=1,fc=2', 0.041729837334383478, 1.7822847]]

#
#[['lr_1.36E-02,conv=1,fc=2', 0.48233429163746566, 0.85388398],
# ['lr_1.34E-02,conv=1,fc=2', 0.1927630967594415, 1.2830796],
# ['lr_1.31E-02,conv=1,fc=2', 0.18637712380434993, 1.2961416],
# ['lr_1.31E-02,conv=1,fc=1', 0.14230569573152105, 1.3963444],
# ['lr_1.32E-02,conv=1,fc=1', 0.087985400668167124, 1.5590332]]




#[['lr_1.35E-03,conv=1,fc=1', 0.40615339010974083, 0.94922304],
# ['lr_1.34E-03,conv=1,fc=1', 0.35460690915158855, 1.018207],
# ['lr_1.39E-03,conv=1,fc=1', 0.31579213420458491, 1.0736252],
# ['lr_1.37E-03,conv=1,fc=1', 0.19132050271853918, 1.2860036],
# ['lr_1.31E-03,conv=1,fc=1', 0.1683968278439098, 1.334703]]

#[['lr_1.34E-04,conv=1,fc=1', 0.66630086409043543, 0.63719225],
# ['lr_1.37E-04,conv=1,fc=1', 0.61832283082111927, 0.69335747],
# ['lr_1.35E-04,conv=1,fc=1', 0.5335746798683717, 0.79256308],
# ['lr_1.33E-04,conv=1,fc=1', 0.50346904057873576, 0.82839185],
# ['lr_1.38E-04,conv=1,fc=1', 0.43005912823390252, 0.9186036]]

# Test top results:
#[['lr_1.35E-05,conv=1,fc=1', 0.34619635470396237, 1.0299268],
# ['lr_1.39E-05,conv=1,fc=1', 0.27503888605243998, 1.1361526],
# ['lr_1.34E-05,conv=1,fc=1', 0.25068886152979908, 1.1762409],
# ['lr_1.32E-05,conv=1,fc=1', 0.23554916114198396, 1.2024291],
# ['lr_1.33E-05,conv=1,fc=1', 0.20501052409356457, 1.2588463]]




#[['lr_1.35E-02,conv=1,fc=2', 0.77906215421419045, 0.49966434], <--
# ['lr_3.35E-04,conv=1,fc=1', 0.74328703474961089, 0.54467696],
# ['lr_3.20E-04,conv=1,fc=1', 0.72439242967910145, 0.56782216],
# ['lr_2.55E-04,conv=1,fc=1', 0.67473585995895935, 0.62724316],
# ['lr_5.55E-04,conv=1,fc=1', 0.66434148167479923, 0.63949901],
# ['lr_3.45E-04,conv=1,fc=1', 0.66229399888408336, 0.64190787],
# ['lr_6.20E-04,conv=1,fc=1', 0.63974350725865958, 0.66834718],
# ['lr_9.55E-04,conv=1,fc=1', 0.63537920339565968, 0.6734488],
# ['lr_1.10E-02,conv=1,fc=2', 0.63514991325416104, 0.67371672],
# ['lr_6.70E-04,conv=1,fc=1', 0.62359447931065692, 0.68720812],
# ['lr_4.70E-04,conv=1,fc=1', 0.62322537765907982, 0.68763876],
# ['lr_9.00E-04,conv=1,fc=1', 0.61966775968534127, 0.69178885],
# ['lr_5.00E-04,conv=1,fc=1', 0.61616954111010125, 0.69586861],
# ['lr_4.90E-04,conv=1,fc=1', 0.61478144821842717, 0.69748724],
# ['lr_1.25E-04,conv=1,fc=1', 0.61150135311833398, 0.70131171],
# ['lr_8.95E-04,conv=1,fc=1', 0.6001435432473986, 0.7145533],
# ['lr_5.65E-04,conv=1,fc=1', 0.5966727709593308, 0.71860033],
# ['lr_6.05E-04,conv=1,fc=1', 0.59076411494688197, 0.72549188],
# ['lr_4.20E-04,conv=1,fc=1', 0.57919431975194235, 0.73899746],
# ['lr_2.60E-04,conv=1,fc=1', 0.57199949918158222, 0.74740696],
# ['lr_5.75E-04,conv=1,fc=1', 0.56873392393420152, 0.75122738],
# ['lr_8.55E-04,conv=1,fc=1', 0.5638483734487999, 0.75694776],
# ['lr_4.25E-04,conv=1,fc=1', 0.55991290646682068, 0.76156026],
# ['lr_2.40E-04,conv=1,fc=1', 0.55653261420000366, 0.76552564],
# ['lr_5.80E-04,conv=1,fc=1', 0.55471002507427714, 0.76766515]]



#[['lr_1.35E-02,conv=1,fc=2', 0.7570571520596624, 0.52755713],
# ['lr_3.20E-04,conv=1,fc=1', 0.61922934405475372, 0.6923002],
# ['lr_3.35E-04,conv=1,fc=1', 0.39884446451469141, 0.95874071],
# ['lr_1.45E-02,conv=1,fc=2', 0.25654917750775563, 1.1663768],
# ['lr_1.43E-02,conv=1,fc=2', 0.14906699455192879, 1.3796229],
# ['lr_1.43E-02,conv=1,fc=1', 0.067371087316504888, 1.6424187],
# ['lr_1.35E-02,conv=1,fc=1', 0.033931309263013895, 1.8394067],
# ['lr_1.45E-02,conv=1,fc=1', 0.00021205298015409976, 2.90838],
# ['lr_3.35E-04,conv=1,fc=2', 3.0920333729552659e-11, 4.9193096],
# ['lr_3.20E-04,conv=1,fc=2', 2.4462258358408561e-81, 13.624054]]


# adam
# https://stats.stackexchange.com/questions/184448/difference-between-gradientdescentoptimizer-and-adamoptimizer-tensorflow
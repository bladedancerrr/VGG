#'same' padding in layers to pad evenly right and left
#using Momentum Optimizer



tf.reset_default_graph()

l2_scale = 5 * (10 ** -4) #L2 regularisation

tf.global_variables_initializer()

with tf.name_scope("input_data"):
  input_data = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name ='input')
  labels     = tf.placeholder(dtype=tf.int64,   shape=[None,], name = 'labels')
  
with tf.name_scope("L1"):
    net = tf.layers.conv2d(inputs = input_data,
                           filters = 64, kernel_size = (3,3), padding = 'same',
                           activation=tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv1')
    
    net = tf.layers.conv2d(inputs = net, filters = 64, kernel_size = (3,3), padding = 'same',
                           activation=tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv2')
    
    net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2, name ='pool')
    
with tf.name_scope("L2"):
    net = tf.layers.conv2d(inputs = net,
                           filters = 128, kernel_size = (3,3), padding = 'same',
                           activation=tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv3')

    net = tf.layers.conv2d(inputs = net,
                           filters = 128, kernel_size = (3,3), padding = 'same',
                           activation=tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv4')
     
    net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2, name ='pool')

with tf.name_scope("L3"):
      net = tf.layers.conv2d(inputs = net,
                           filters = 256, kernel_size = (3,3), padding = 'same',
                           activation=tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv5')
    
      net = tf.layers.conv2d(inputs = net,
                           filters = 256, kernel_size = (3,3), padding = 'same',
                           activation=tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv6')
      
      net = tf.layers.conv2d(inputs = net,
                           filters = 256, kernel_size = (3,3), padding = 'same',
                           activation=tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv7')
      
      
      net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2, name ='pool')
      
with tf.name_scope("L4"):
  
  
      net = tf.layers.conv2d(inputs = net,
                           filters = 512, kernel_size = (3,3), padding = 'same',
                           activation=tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv8')
    
      net = tf.layers.conv2d(inputs = net,
                           filters = 512, kernel_size = (3,3), padding = 'same',
                           activation=tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv9')
      
      
      net = tf.layers.conv2d(inputs = net,
                           filters = 512, kernel_size = (3,3), padding = 'same',
                           activation=tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv10')
      
      net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2, name ='pool')

      
with tf.name_scope("L5"):
      net = tf.layers.conv2d(inputs = net,
                           filters = 512, kernel_size = (3,3), padding = 'same',
                           activation=tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv11')
    
      net = tf.layers.conv2d(inputs = net,
                           filters = 512, kernel_size = (3,3), padding = 'same',
                           activation=tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv12')
    
      net = tf.layers.conv2d(inputs = net,
                           filters = 512, kernel_size = (3,3), padding = 'same',
                           activation=tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv13')
    
      net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2, name ='pool')
    
with tf.name_scope("L6"):
  
      net = tf.layers.conv2d(inputs = net,
                           filters = 10, kernel_size = (1,1),
                           kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale), 
                           name = 'conv14')
      net = tf.reshape(net, [-1,10], name ='reshaped')
      predictions = tf.nn.softmax(net)
     
      pred_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = predictions))
                                                                    
    
      l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
      loss = pred_loss + l2_loss 
      tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
      tf.summary.scalar("loss", loss)
      accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, axis = 1),
                                                 labels),
                                        tf.float32))
      tf.summary.scalar("accuracy", accuracy)

      
      
      
      
      


    
    

      
      
      
      

      
      



    
      
   
   

    







from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import inspect
import tensorflow as tf

from scripts.generate_dataset import GO_ID

sequence_loss = tf.contrib.legacy_seq2seq.sequence_loss
layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope


class CSMN(object):
    def _embedding_to_hidden(self, inp, size, scope = 'Wh', reuse = True):
        with arg_scope([layers.fully_connected],
                       num_outputs = self.mem_dim,
                       activation_fn = tf.nn.relu,
                       weights_initializer = self.w_initializer,
                       biases_initializer = self.b_initializer):
            output2dim = tf.reshape(layers.fully_connected(inp, reuse = reuse, scope = scope),
                                    [-1, size, self.mem_dim])
            return output2dim
    
    def _init_img_mem(self, conv_cnn):
        '''
        
        '''
        img_mem_a = self._embedding_to_hidden(conv_cnn,
                                              1,
                                              scope = 'Wima',
                                              reuse = False)
        img_mem_c = self._embedding_to_hidden(conv_cnn,
                                              1,
                                              scope = 'Wimc',
                                              reuse = False)
        
        return img_mem_a, img_mem_c
    
    def text_cnn(self, inp, filter_sizes, mem_size, scope):
        pooled_outputs = []
        with arg_scope([layers.conv2d],
                       stride = 1,
                       padding = 'VALID',
                       activation_fn = tf.nn.relu,
                       weights_initializer = self.w_initializer,
                       biases_initializer = self.b_initializer):
            for j, filter_size in enumerate(filter_sizes):
                conv = layers.conv2d(inp, self.num_channels, [filter_size, self.mem_dim],
                                     scope = scope + '-conv%s' % filter_size)
                pooled = layers.max_pool2d(conv, [mem_size - filter_size + 1, 1],
                                           stride = 1, padding = 'VALID',
                                           scope = scope + '-pool%s' % filter_size)
                pooled_outputs.append(pooled)
        
        return pooled_outputs
    
    def _init_words_mem(self, words, shape, words_mask, max_length, is_first_time, is_init_B):
        emb_A = tf.nn.embedding_lookup(self.Wea, words)
        emb_C = tf.nn.embedding_lookup(self.Wec, words)
        
        tiled_word_mask = tf.to_float(tf.tile(tf.reshape(words_mask, [self.batch_size, max_length, 1]), \
                                             [1, 1, self.mem_dim]))
        
        words_mem_A = self._embedding_to_hidden(emb_A, max_length, reuse = not is_first_time) * tiled_word_mask
        words_mem_C = self._embedding_to_hidden(emb_C, max_length) * tiled_word_mask
        
        if is_init_B:
            emb_B = tf.nn.embedding_lookup(self.Web, words)
            words_mem_B = self._embedding_to_hidden(emb_B, max_length) * tiled_word_mask
            return words_mem_A, words_mem_B, words_mem_C
            
        
        return words_mem_A, words_mem_C
        
    
    def __init__(self, inputs, config, name = 'CSNM', is_training = True):
        
        #Set config varaibles as model's variable.
        attrs = { k:v for k,v in inspect.getmembers(config) \
                 if not k.startswith('__') and not callable(k) }
        
        for attr in attrs:
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(config, attr))
        
        self.is_training = is_training
        self.name = name
        
        #Initialize w & b
        self.w_initializer = tf.glorot_uniform_initializer()
        self.b_initializer = tf.constant_initializer(0.0)
        
        conv_cnn = inputs['img_embed']        
        output_largest_length = inputs['caption_len']
        caption = inputs['caption_id']
        answer = inputs['answer_id']
        output_mask = inputs['caption_mask']
        output_largest_length = tf.reduce_max(output_largest_length)
        
        if self.use_user_context:
            context = inputs['context_id']
            context_largest_length = inputs['context_len']
            context_mask = inputs['context_mask']
            context_largest_length = tf.reduce_max(context_largest_length)
        
        emb_shape = [self.vocab_size, self.word_emb_dim]
        
        #Input embedding
        self.Wea = tf.get_variable("Wea", shape = emb_shape, \
                                   initializer = self.w_initializer)
        #Embedding for output caption
        self.Web = tf.get_variable("Web", shape = emb_shape, \
                                   initializer = self.w_initializer)
        
        #Output embedding
        self.Wec = tf.get_variable("Wec", shape = emb_shape, \
                                   initializer = self.w_initializer)
        
        #Output word weights
        self.Wf = tf.get_variable("Wf", shape = [self.num_channels_total, self.vocab_size], \
                                  initializer = self.w_initializer)
        
        #Output word bias
        self.bf = tf.get_variable("bf", shape = [self.vocab_size], \
                                  initializer = self.b_initializer)
        
        
        #Image memeory. Eqn (1) and (2)
        img_mem_A, img_mem_C = self._init_img_mem(conv_cnn)
        
        if self.use_user_context:
            context_mem_A, context_mem_C = self._init_words_mem(context, 
                                                                tf.stack([self.batch_size, context_largest_length]),
                                                                context_mask,
                                                                self.max_context_length,
                                                                is_first_time = True,
                                                                is_init_B = False)
        
        if self.is_training:
            output_mem_A, output_mem_B, output_mem_C = self._init_words_mem(caption,
                                                                         tf.stack([self.batch_size, output_largest_length]),
                                                                         output_mask,
                                                                         self.max_output_length,
                                                                         is_first_time = (not self.use_user_context) or False,
                                                                         is_init_B = True)
        else:
            output_mem_A, output_mem_B, output_mem_C = self.Wea, self.Web, self.Wec
        
        
        def _loop_condition(iterator, *args):
            if self.is_training:
                return tf.less(iterator, output_largest_length)
            else:
                return tf.less(iterator, self.max_output_length)
        
        def _loop_body(iterator, out_words_array, out_mem_state_A, out_mem_state_C,
                      out_mem_A, out_mem_B, out_mem_C):
            def train_input():
                out_A_slice = tf.slice(out_mem_A,
                                       [0, iterator, 0],
                                       [self.batch_size, 1, self.mem_dim])
                out_C_slice = tf.slice(out_mem_C,
                                       [0, iterator, 0],
                                       [self.batch_size, 1, self.mem_dim])
                query = tf.slice(out_mem_B,
                                 [0, iterator, 0],
                                 [self.batch_size, 1, self.mem_dim])
                return out_A_slice, out_C_slice, query
            
            def test_input():
                def go_symbol():
                    return tf.constant(GO_ID, shape = [self.batch_size, 1], dtype = tf.iint32)
                
                def not_go_symbol():
                    out = out_words_array.read(iterator - 1)
                    out.set_shape([self.batch_size, self.num_channels_total])
                    out = tf.matmul(out, self.Wf) + self.bf
                    out = tf.reshape(tf.to_int32(tf.argmax(tf.nn.softmax(out), 1)),
                                     [self.batch_size, 1])
                    return out
                
                prev_word = tf.cond(tf.equal(iterator, 0), go_symbol, not_go_symbol)
                
                out_A_slice = self._embedding_to_hidden(tf.nn.embedding_lookup(out_mem_A, prev_word), 1)
                
                out_C_slice = self._embedding_to_hidden(tf.nn.embedding_lookup(out_mem_C, prev_word), 1)
                
                query = self._embedding_to_hidden(tf.nn.embedding_lookup(out_mem_B, prev_word), 1)
                
                return out_A_slice, out_C_slice, query
            
            out_A_slice, out_C_slice, query = train_input() if self.is_training else test_input()
            
            #Generate input vector at time t. Eqn 8.
            query = self._embedding_to_hidden(query, 
                                              1,
                                              scope = "Wq",
                                              reuse = False)
             
            out_Ai_slice = tf.slice(out_mem_state_A, 
                                    [0, 0, 0],
                                    [self.batch_size, iterator, self.mem_dim])
            out_Ci_slice = tf.slice(out_mem_state_C,
                                    [0, 0, 0],
                                    [self.batch_size, iterator, self.mem_dim])
            
            out_padding = tf.tile(tf.constant(0.0, shape = [self.batch_size, 1, self.mem_dim]),
                                  [1, self.max_output_length - iterator - 1, 1])
            
            out_mem_state_A = tf.concat([out_Ai_slice, out_A_slice, out_padding], 1)
            
            out_mem_state_C = tf.concat([out_Ci_slice, out_C_slice, out_padding], 1)
            
            out_mem_state_A = tf.reshape(out_mem_state_A,
                                         [self.batch_size, self.max_output_length, self.mem_dim])
            out_mem_state_C = tf.reshape(out_mem_state_C,
                                         [self.batch_size, self.max_output_length, self.mem_dim])
            
            if self.use_user_context:
                mem_A = tf.concat([img_mem_A, context_mem_A, out_mem_state_A], 1)
            else:
                mem_A = tf.concat([img_mem_A, out_mem_state_A], 1)
            
            #Eqn 9
            p_t_inner = tf.matmul(query, mem_A, adjoint_b = True)
            
            if self.use_user_context:
                memory_sizes = [self.img_memory_size, self.max_context_length, self.max_output_length]
            else:
                memory_sizes = [self.img_memory_size, self.max_output_length]
                
            p_t_inner_reshaped = tf.reshape(p_t_inner, [-1, self.memory_size])
            p_t = tf.nn.softmax(p_t_inner_reshaped, name = 'attention')
            
            if self.use_user_context:
                img_attn, context_attn, out_attn = tf.split(p_t, memory_sizes, axis = 1)
            else:
                img_attn, out_attn = tf.split(p_t, memory_sizes, axis = 1)
            
            img_attn = tf.tile(tf.reshape(img_attn, [self.batch_size, self.img_memory_size, 1]),
                                          [1, 1, self.mem_dim])
            if self.use_user_context:
                context_attn = tf.tile(tf.reshape(context_attn, [self.batch_size, self.max_context_length, 1]),
                                                  [1, 1, self.mem_dim])
            
            out_attn = tf.tile(tf.reshape(out_attn, [self.batch_size, self.max_output_length, 1]),
                                          [1, 1, self.mem_dim])
            
            img_weighted_mem_C = tf.reshape(img_mem_C * img_attn, [self.batch_size, self.mem_dim])
            
            if self.use_user_context:
                context_weighted_mem_C = tf.expand_dims(context_mem_C * context_attn, -1)
            
            out_weighted_mem_C = tf.expand_dims(out_mem_state_C *out_attn, -1)

            #Eqn 10            
            pooled_outputs = []
            if self.use_user_context:
                pooled_outputs += self.text_cnn(context_weighted_mem_C,
                                                self.context_filter_sizes,
                                                self.max_context_length,
                                                scope = 'context')
            
            pooled_outputs += self.text_cnn(out_weighted_mem_C,
                                            self.output_filter_sizes, 
                                            self.max_output_length,
                                            scope = 'output')
            
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, self.num_channels_total])
            
            #Eqn 11
            with arg_scope([layers.fully_connected],
                           num_outputs = self.num_channels_total,
                           activation_fn = tf.nn.relu,
                           weights_initializer = self.w_initializer,
                           biases_initializer = self.b_initializer):
                img_feature = layers.fully_connected(img_weighted_mem_C,
                                                     reuse = False,
                                                     scope = 'Wpool5')
                img_added_result = img_feature + h_pool_flat
                out = layers.fully_connected(img_added_result, 
                                             reuse = False,
                                             scope = 'Wo')
            out_words_array = out_words_array.write(iterator, out)
            
            return (iterator + 1, out_words_array, out_mem_state_A, out_mem_state_C,
                    out_mem_A, out_mem_B, out_mem_C)
            
        ## Initialize while loop variables
        iterator = tf.constant(0, dtype = tf.int32)
        
        out_words_array = tf.TensorArray(dtype = tf.float32, size = 0, clear_after_read = False, dynamic_size = True)
        
        out_mem_state_A = tf.constant(0.0, dtype = tf.float32, 
                                      shape = [self.batch_size, self.max_output_length, self.mem_dim])
        out_mem_state_C = tf.constant(0.0, dtype = tf.float32,
                                      shape = [self.batch_size, self.max_output_length, self.mem_dim])
        
        loop_vars = [iterator,
                     out_words_array,
                     out_mem_state_A,
                     out_mem_state_C,
                     output_mem_A,
                     output_mem_B,
                     output_mem_C]
        
        loop_outputs = tf.while_loop(_loop_condition, _loop_body, loop_vars, back_prop = True, parallel_iterations = 1)
        
        sequence_outputs = loop_outputs[1].stack()
        sequence_outputs.set_shape([None, self.batch_size, self.num_channels_total])
        
        self.om_s_a = loop_outputs[2]
        sequence_outputs = tf.transpose(sequence_outputs, perm = [1, 0, 2])
        sequence_outputs = tf.reshape(sequence_outputs, [-1, self.num_channels_total])
        
        #Eqn 12
        final_outputs = tf.matmul(sequence_outputs, self.Wf) + self.bf
        out_probs = tf.nn.softmax(final_outputs)
        self.prob = tf.reshape(out_probs, [self.batch_size, -1, self.vocab_size])
        self.argmax = tf.argmax(self.prob, 2)
        
        output_mask = tf.slice(output_mask,
                            [0, 0],
                            [self.batch_size, output_largest_length])
        
        answer = tf.slice(answer, [0, 0], [self.batch_size, output_largest_length])
        self.loss = sequence_loss([final_outputs],
                                  [tf.reshape(answer, [-1])],
                                  [tf.reshape(tf.to_float(output_mask), [-1])])
        
        out_words_array.close()
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
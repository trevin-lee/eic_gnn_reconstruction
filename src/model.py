import os
import time
from graph_nets.graphs      import GraphsTuple
from graph_nets             import utils_tf
from typing                 import Generator, Tuple

import numpy                as np
import tensorflow           as tf
import logging
import tqdm

from config_loader          import ConfigLoader
from exceptions             import ModelException
from data_generator         import DataGenerator



class Model:

    """
    This class handles all operations relevant to the model. The config
    class must be passed in to instantiate a class. Then, the optimizer,
    loss function, and model must be set through the set functions
    available below. Then the train function can be called to train
    the model. 
    """

    def __init__(
            self, 
            config_loader: ConfigLoader, 
            model: tf.keras.models,
            val_data: DataGenerator,
            train_data: DataGenerator
    ):
        
        config = self.config = config_loader

        self.val_data = val_data
        self.train_data = train_data
        self.model = model

        "Setup checkpointing"
        self.checkpoint = tf.train.Checkpoint(module=self.model)

        "Automatically determine model parameters"
        self._set_regression_loss_fn()
        self._set_optimizer()


        "Initialize logger for error warnings, and info"
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.logger = logging.getLogger(__name__)


    def train_model(self):
        config = self.config
        
        training_loss_epoch = []
        val_loss_epoch = []
        curr_loss = 1e5


        checkpoint = tf.train.Checkpoint(module=self.model)
        best_ckpt_prefix = os.path.join(config.RESULT_DIR_PATH, 'best_model') # prefix.

        best_ckpt = tf.train.latest_checkpoint(config.RESULT_DIR_PATH)
        last_ckpt_path = config.RESULT_DIR_PATH + '/last_saved_model'
    
        
        #Main Epoch Loop
        for epoch in range(config.NUM_EPOCHS):
            print('\n\nStarting epoch: {}'.format(epoch))
            epoch_start = time.time()

            training_loss = []
            val_loss = []

            # Train
            print('Training...')
            i = 0
            start = time.time()
            for graph_data_tr, targets_tr, _ in self._get_batch(self.train_data.generator()):

                losses_tr = self._train_step(graph_data_tr, targets_tr)
                training_loss.append(losses_tr.numpy())

                if not (i)%100:
                    end = time.time()
                    print('Iter: {:03d}, Tr_loss_curr: {:.4f}, Tr_loss_mean: {:.4f}'. \
                        format(i, training_loss[-1], np.mean(training_loss)), end='  ')
                    print('Took {:.3f} secs'.format(end-start))
                    start = time.time()

                i += 1 

            end = time.time()
            print('Iter: {:03d}, Tr_loss_curr: {:.4f}, Tr_loss_mean: {:.4f}'. \
                format(i, training_loss[-1], np.mean(training_loss)), end='  ')
            print('Took {:.3f} secs'.format(end-start))

            training_loss_epoch.append(training_loss)
            training_end = time.time()

            # validate
            print('\nValidation...')
            i = 1
            all_targets = []
            all_outputs = []
            all_etas = []
            all_meta = []
            start = time.time()

            for graph_data_val, targets_val, meta_val in self._get_batch(self.val_data.generator()):
                losses_val, output_vals = self._val_step(graph_data_val, targets_val)
                targets_val = targets_val.numpy()
                output_vals = output_vals.numpy().squeeze()

                val_loss.append(losses_val.numpy())
                all_targets.append(targets_val)
                all_outputs.append(output_vals)
                all_meta.append(meta_val)

                if not (i)%100:
                    end = time.time()
                    print('Iter: {:03d}, Val_loss_curr: {:.4f}, Val_loss_mean: {:.4f}'. \
                        format(i, val_loss[-1], np.mean(val_loss)), end='  ')
                    print('Took {:.3f} secs'.format(end-start))
                    start = time.time()

                i += 1 

            end = time.time()
            print('Iter: {:03d}, Val_loss_curr: {:.4f}, Val_loss_mean: {:.4f}'. \
                format(i, val_loss[-1], np.mean(val_loss)), end='  ')
            print('Took {:.3f} secs'.format(end-start))

            epoch_end = time.time()

            all_targets = np.concatenate(all_targets)
            all_outputs = np.concatenate(all_outputs)
            all_meta = np.concatenate(all_meta)

            val_loss_epoch.append(val_loss)

            np.savez(config.RESULT_DIR_PATH+'/losses', training=training_loss_epoch, validation=val_loss_epoch)
            checkpoint.write(last_ckpt_path)

            val_mins = int((epoch_end - training_end)/60)
            val_secs = int((epoch_end - training_end)%60)
            training_mins = int((training_end - epoch_start)/60)
            training_secs = int((training_end - epoch_start)%60)
            print('\nEpoch {} ended\nTraining: {:2d}:{:02d}\nValidation: {:2d}:{:02d}'. \
                format(epoch, training_mins, training_secs, val_mins, val_secs))

            if np.mean(val_loss)<curr_loss:
                print('\nLoss decreased from {:.6f} to {:.6f}'.format(curr_loss, np.mean(val_loss)))
                print('Checkpointing and saving predictions to:\n{}'.format(config.RESULT_DIR_PATH))
                curr_loss = np.mean(val_loss)
                np.savez(config.RESULT_DIR_PATH+'/predictions', 
                        targets=all_targets, 
                        outputs=all_outputs,
                        meta=all_meta)
                checkpoint.save(best_ckpt_prefix)
            else:
                print('\nLoss did not decrease from {:.6f}'.format(curr_loss))

            if not (epoch+1)%5:
                self.optimizer.learning_rate = self.optimizer.learning_rate/2
                if self.optimizer.learning_rate<1e-6:
                    self.optimizer.learning_rate = 1e-6 
                    print('\nLearning rate would fall below 1e-6, setting to: {:.5e}'.format(self.optimizer.learning_rate.value()))
                else:
                    print('\nLearning rate decreased to: {:.5e}'.format(self.optimizer.learning_rate.value()))



    """
    These functions can be called to set the loss function and optimizer.
    If not called the default optimizer and loss function will be used 
    (MAE and Adam). 
    """

    def _set_optimizer(self):
        config = self.config

        match config.OPTIMIZER.lower():
            case 'adam':
                self.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=config.LEARNING_RATE
                )
            case 'sgd':
                self.optimizer = tf.keras.optimizers.SGD(
                    learning_rate=config.LEARNING_RATE
                )
            case 'rmsprop':
                self.optimizer = tf.keras.optimizers.RMSprop(
                    learning_rate=config.LEARNING_RATE 
                )
            case 'adagrad':
                self.optimizer = tf.keras.optimizers.Adagrad(
                    learning_rate=config.LEARNING_RATE
                )
            case 'adamax':
                self.optimizer = tf.keras.optimizers.Adamax(
                    learning_rate=config.LEARNING_RATE
                )
            case 'nadam':
                self.optimizer = tf.keras.optimizers.Nadam(
                    learning_rate=config.LEARNING_RATE 
                )
            case _:
                raise ModelException(
                    f"Unknown optimizer: {config.OPTIMIZER}"
                )


    def _set_regression_loss_fn(self, ):
        config = self.config

        match config.LOSS_FUNCTION.lower():
            case 'mae':
                self.regression_loss_function = (
                    tf.keras.losses.MeanAbsoluteError()
                )
            case 'mse':
                self.regression_loss_function = (
                    tf.keras.losses.MeanSquaredError()
                )
            case 'huber':
                self.regression_loss_function = (
                    tf.keras.losses.Huber()
                )
            case 'mape':
                self.regression_loss_function = (
                    tf.keras.losses.MeanAbsolutePercentageError()
                )
            case 'msle':
                self.regression_loss_function = (
                    tf.keras.losses.MeanSquaredLogarithmicError()
                )
            case 'log_cosh':
                self.regression_loss_function = (
                    tf.keras.losses.LogCosh()
                )
            case _:
                raise ModelException(
                    f"Unknown loss function: {config.LOSS_FUNCTION}"
                )

    

    """
    The below functions are the private helper functions responsible for
    training the model. 
    """

    def _get_input_signature(self) -> list:
        samp_graph, samp_target, samp_meta = next(self._get_batch(self.train_data.generator()))
        self.train_data.kill_processes()
        graph_spec = utils_tf.specs_from_graphs_tuple(samp_graph, True, True, True)

        match self.config.OUTPUT_DIMENSIONS:
            case 1:
                self.provided_shape = [None,]
            case 2:
                self.provided_shape = [None, None]
            case 3:
                self.provided_shape = [None, None, None]
            case _:
                raise ModelException(
                    f"Unsupported OUTPUT_DIMENSION: {self.config.OUTPUT_DIMENSION}"
                )
        return [
            graph_spec,
            tf.TensorSpec(
                shape=self.provided_shape,
                dtype=tf.float32
            )
        ]


    def _train_step(self, graphs, targets) -> float:
        @tf.function(input_signature=self._get_input_signature())
        def _wrapped_train_step(graphs, targets):
            model = self.model
            
            with tf.GradientTape() as tape:
                predictions = model(graphs).globals
                loss = self.regression_loss_fn(targets, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss
        
        return _wrapped_train_step(graphs, targets)


    def _val_step(self, graphs, targets) -> Tuple[float, tf.Tensor]:
        @tf.function(input_signature=self._get_input_signature())
        def _wrapped_val_step(graphs, targets):
            model = self.model
            
            predictions = model(graphs).globals
            loss = self.regression_loss_fn(targets, predictions)

            return loss, predictions
        
        return _wrapped_val_step(graphs, targets)


    def _get_batch(self, data_iter) -> Generator[Tuple, Tuple, any]:
        """     
        data_iter is a tuple of lists
        list of graphs, list of targets, list of meta data with
        Each entry of the lists has info for one event in the batch
        
         targets structure: 
            For 1D: Just a list [ genP0, genP1, genP2, ...]
            For 2D: list of tuples [ (genP0, gentheta0),...]
            Convert targets to tf.tensor
            1D shape (len(targets), ), i.e. [ genP0, genP1, genP2, ...]
            2D shape (len(targets), 2), i.e. [ [genP0, gentheta0],  ...]
        """

        for graphs, targets, meta in data_iter:

            graphs = self._convert_to_tuple(graphs)
            targets = tf.convert_to_tensor(targets, dtype=tf.float32)
            yield graphs, targets, meta


    def _convert_to_tuple(self, graphs: list) -> GraphsTuple:
        nodes = []
        edges = []
        globals = []
        senders = []
        receivers = []
        n_node = []
        n_edge = []
        offset = 0

        for graph in graphs:
            nodes.append(graph['nodes'])
            globals.append([graph['globals']])
            n_node.append([len(graph['nodes'])])

            if graph['senders'] is not None:
                senders.append(graph['senders'] + offset)
            if graph['receivers'] is not None:
                receivers.append(graph['receivers'] + offset)
            if graph['edges'] is not None:
                edges.append(graph['edges'])
                n_edge.append([len(graph['edges'])])
            else:
                n_edge.append([0])

            offset += len(graph['nodes'])

        nodes = tf.convert_to_tensor(np.concatenate(nodes), dtype=tf.float32)
        globals = tf.convert_to_tensor(np.concatenate(globals), dtype=tf.float32)
        n_node = tf.convert_to_tensor(np.concatenate(n_node), dtype=tf.int64)
        n_edge = tf.convert_to_tensor(np.concatenate(n_edge), dtype=tf.int64)

        if senders:
            senders = tf.convert_to_tensor(np.concatenate(senders), dtype=tf.int32)
        else:
            senders = tf.convert_to_tensor([], dtype=tf.int32)

        if receivers:
            receivers = tf.convert_to_tensor(np.concatenate(receivers), dtype=tf.int32)
        else:
            receivers = tf.convert_to_tensor([], dtype=tf.int32)

        if edges:
            edges = tf.convert_to_tensor(np.concatenate(edges), dtype=tf.float32)
        else:
            edges = tf.convert_to_tensor([], dtype=tf.float32)
            edges = tf.reshape(edges, (-1, 1))

        graph = GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=globals,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge
        )

        return graph
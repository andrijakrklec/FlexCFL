import numpy as np
import tensorflow as tf

'''
Define the base actor of federated learning framework
like Server, Group, Client.
'''
class Actor(object):
    def __init__(self, id, actor_type, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}, model=None):
        self.id = id
        self.train_data = train_data
        self.test_data = test_data
        self.model = model # callable tf.keras.model
        self.actor_type = actor_type
        self.name = 'NULL'
        self.latest_params, self.latest_updates = None, None
        # init train and test size to zero, it will depend on the actor type
        self.train_size, self.test_size = 0, 0 
        self.uplink, self.downlink = [], [] # init to empty, depend on the actor type
        # Is this actor can train or test,
        # Note: This variable have differenct meaning according to differnent type of actor
        self.trainable, self.testable = False, False 

        self.preprocess()

    def preprocess(self):
        self.name = str(self.actor_type) + str(self.id)
        self.latest_params = self.get_params()
        self.latest_updates = [np.zeros_like(ws) for ws in self.latest_params]

    def get_params(self):
        if self.model:
            return self.model.get_weights()
    
    def set_params(self, weights):
        # Set the params of model,
        # But the latest_params and latest_updates will not be refreshed
        if self.model:
            self.model.set_weights(weights)
            
    def solve_gradients(self, num_epoch=1, batch_size=10):
        '''
        Solve the local optimization base on local training data, 
        the gradient is NOT applied to model
        
        Return: num_samples, Gradients
        '''
        if self.train_data['y'] > 0:
            X, y_true = self.train_data['x'], self.train_data['y']
            num_samples = y_true.shape[0]
            with tf.GradientTape() as tape:
                y_pred = self.model(X)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            
            return num_samples, gradients
        else:
            # Return 0 and all zero gradients [0, 0, ...],
            # if this actor has not training set
            return 0, [np.zeros_like(ws) for ws in self.latest_updates]

    def solve_inner(self, num_epoch=1, batch_size=10):
        '''
        Solve the local optimization base on local training data,
        This Function will not change the params of model,
        Call apply_update() to change model
        
        Return: num_samples, train_acc, train_loss, update
        '''
        if self.train_data['y'].shape[0] > 0:
            X, y_true = self.train_data['x'], self.train_data['y']
            num_samples = y_true.shape[0]
            # Confirm model params is euqal to latest params
            t0_weights = self.get_params()
            # Use model.fit() to train model
            history = self.model.fit(X, y_true, batch_size, num_epoch, verbose=0)
            t1_weights = self.get_params()
            
            # Roll-back the weights of model
            self.set_params(t0_weights)
            # Calculate the updates
            update = [(w1-w0) for w0, w1 in zip(t0_weights, t1_weights)]
            # Get the train accuracy and train loss
            #print(history.history) # Debug
            train_acc = history.history['accuracy']
            train_loss = history.history['loss']
            #print(train_acc) # Debug
            
            return num_samples, train_acc, train_loss, update
        else:
            # Return 0,0,0 and all zero updates [0, 0, ...],
            # if this actor has not training set
            return 0, [0], [0], [np.zeros_like(ws) for ws in self.latest_params]

    def apply_update(self, update):
        '''
        Apply update to model and Refresh the latest_params and latest_updates
        Return:
            1, Latest model params
        '''
        t0_weights = self.get_params()
        t1_weights = [(w0+up) for up, w0 in zip(update, t0_weights)]
        self.set_params(t1_weights)
        # Refresh the latest_params and latest_updates attrs
        self.latest_updates = update
        self.latest_params = t1_weights
        return self.latest_params
    """
    def refresh_latest_params_updates(self):
        '''
        Call this function to refresh the latest_params and latst_updates
        Whenever the model change
        '''
        prev_params = self.latest_params
        latest_params = self.get_params()
        self.latest_updates = [(w1-w0) for w0, w1 in zip(prev_params, latest_params)]
        self.latest_params = latest_params
    """ 
    def test_locally(self):
        '''
        Test the model on local test dataset
        '''
        if self.test_data['y'].shape[0] > 0:
            X, y_true = self.test_data['x'], self.test_data['y']
            loss, acc = self.model.evaluate(X, y_true, verbose=0)
            return self.test_data['y'].shape[0], acc, loss
        else:
            return 0, 0, 0

    def has_uplink(self):
        if len(self.uplink) > 0:
            return True
        return False

    def has_downlink(self):
        if len(self.downlink) > 0:
            return True
        return False

    def add_downlink(self, nodes):
        # Note: The repetitive node is not allow
        self.downlink = list(set(self.downlink + nodes))
        return

    def add_uplink(self, nodes):
        self.uplink = list(set(self.uplink + nodes))
        return
    
    def delete_downlink(self, nodes):
        self.downlink = [c for c in self.downlink if c not in nodes]
        return

    def delete_uplink(self, nodes):
        self.uplink = [c for c in self.uplink - nodes if c not in nodes]
        return

    # Train() and Test() depend on actor type
    def test(self):
        return

    def train(self):
        return


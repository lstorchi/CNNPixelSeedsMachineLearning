# flake8: noqa: E402, F401
"""
Doublet model with hit shapes and info features.
"""

import argparse
import datetime
import json
import tempfile
import os
from dataset import Dataset
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from model_architectures import *
from importlib import reload
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt 

from keras import backend as K
import tensorflow as tf

from random import shuffle

t_now = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
# Model configuration
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="data/")
parser.add_argument('--n_epochs', type=int, default=200,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--log_dir', type=str, default="models/cnn_doublet")
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--maxnorm', type=float, default=10.)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--open', action="store_true")
#parser.add_argument('--', type=float, default=1.0)
args = parser.parse_args()

if args.open:
        dataset.dataLab = dataset.dataLabOpen
        dataset.featureLabs = dataset.featureLabsOpen
        dataset.particleLabs = dataset.particleLabsOpen
        dataset.inHitLabs = dataset.inHitLabsOpen
        dataset.outHitLabs = dataset.outHitLabsOpen
        reload(dataset)
if args.name is None:
    args.name = input('model name: ')

if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)

log_dir_tf = args.log_dir + '/' + args.name
# "/eos/cms/store/cmst3/group/dehep/convPixels/TTBar_13TeV_PU35/"
remote_data = args.data
debug_data  = args.data + "/debug/"
debug_files = [ debug_data + el for el in os.listdir(debug_data)]

print("> Loading data ...")
all_files = [remote_data + el for el in os.listdir(remote_data)]
shuffle(all_files)

train_files = all_files[:int(0.8*len(all_files))]
val_files = all_files[int(0.8*len(all_files)):int(0.9*len(all_files))]
test_files = all_files[int(0.9*len(all_files)):]

if args.debug:
    print( "(debug mode)")
    test_files = all_files[:1]
    train_files = all_files[:1]
    val_files = all_files[:1]
    train_data = Dataset(train_files,numofr =1)
    val_data = train_data
    test_data = train_data
    args.n_epochs = 2
else:
    train_data = Dataset(train_files)
    val_data = Dataset(val_files)
    test_data = Dataset(test_files)


train_data = train_data.balance_data()
val_data = val_data.balance_data()
test_data = test_data

#X_hit, X_info, y = train_data.get_layer_map_data()
#X_val_hit, X_val_info, y_val = val_data.get_layer_map_data()
#X_test_hit, X_test_info, y_test = test_data.get_layer_map_data()

X_hit, X_info, y = train_data.get_data(angular_correction=False)
X_val_hit, X_val_info, y_val = val_data.get_data(angular_correction=False)
X_test_hit, X_test_info, y_test = test_data.get_data(angular_correction=False)

print("Training size: " + str(X_hit.shape[0]))
print("Val size: " + str(X_val_hit.shape[0]))
print("Test size: " + str(X_test_hit.shape[0]))

train_input_list = [X_hit, X_info]
val_input_list = [X_val_hit, X_val_info]
test_input_list = [X_test_hit, X_test_info]

print(train_input_list[0].shape)
print(val_input_list[0].shape)
print(test_input_list[0].shape)

#model = small_doublet_model(args, train_input_list[0].shape[-1],train_input_list[1].shape[-1])
model = dense_model(args, train_input_list[0].shape[-1],train_input_list[1].shape[-1])

if args.verbose:
    model.summary()

print('> Training')

fname = args.log_dir + "/" + args.name
with open(fname + ".json", "w") as outfile:
    json.dump(model.to_json(), outfile)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=args.patience),
    ModelCheckpoint(fname + ".h5", save_best_only=True,
                    save_weights_only=True),
    TensorBoard(log_dir=log_dir_tf, histogram_freq=0,
                write_graph=True, write_images=True)
]


model.fit(train_input_list, y, batch_size=args.batch_size, epochs=args.n_epochs, shuffle=True,
          validation_data=(val_input_list, y_val), callbacks=callbacks, verbose=args.verbose)

model.load_weights(fname + ".h5")

loss, acc = model.evaluate(test_input_list, y_test, batch_size=args.batch_size)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

y_pred_keras = model.predict(test_input_list)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test[:,0], y_pred_keras[:,0])
auc_keras = auc(fpr_keras, tpr_keras)
print('ROC - AUC = {:.4f}'.format(auc_keras))
plt.figure(1)
#plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label=args.name+' (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.grid()
plt.savefig('ROC_'+args.name)
'''
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
'''


print("> Saving model " + fname)
model.save_weights(fname + ".h5", overwrite=True)


frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "./", fname +".pb", as_text=False)
tf.train.write_graph(frozen_graph, "./", fname + ".txt", as_text=True)

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
 
import h5py
def data():

    try:
        hf["target"].shape
    except:
        hf = h5py.File('faces.hdf5','r+')
    num_samples = hf["input"].shape[0]

    print "number of samples in dataset : %i" %num_samples

    split_dict = {
         'train': {'input': (2000, num_samples), 'target': (2000, num_samples)},
         'test': {'input': (0, 1000), 'target': (0, 1000)},
         'val': {'input': (1000, 2000), 'target': (1000, 2000)}
    }
    hf.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    train_set = H5PYDataset('faces.hdf5', which_sets=('train',))
    test_set = H5PYDataset('faces.hdf5', which_sets=('test',))
    val_set = H5PYDataset('faces.hdf5', which_sets=('val',))

    batch_size = 128

#TODO : use shuffledscheme instead?  Seems slower, might have screwed up the chunksize in the HDF5 files?

    tr_scheme = SequentialScheme(examples=train_set.num_examples, batch_size=batch_size)
    tr_stream = DataStream(train_set, iteration_scheme=tr_scheme)

    val_scheme = SequentialScheme(examples=val_set.num_examples, batch_size=batch_size)
    val_stream = DataStream(val_set, iteration_scheme=val_scheme)

    test_scheme = SequentialScheme(examples=test_set.num_examples, batch_size=batch_size)
    test_stream = DataStream(test_set, iteration_scheme=test_scheme)
    hf.close()
    return num_samples, train_set, test_set, val_set, tr_scheme, tr_stream, val_scheme, val_stream, test_scheme, test_stream




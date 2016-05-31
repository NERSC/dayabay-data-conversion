###############################3######
# Sam Kohn 2016
# Extract IBD candidates
####################################333

import os
import sys
from collections import defaultdict
import roottools
import array
import numpy as np
import itertools
import pickle as pkl
import pandas
import h5py
import thread
import logging
__all__ = ['unflattenData']
logging.basicConfig(level=logging.DEBUG)

METADATA_NAMES = (
        'runno',
        'fileno', 
        'site',
        'det',
        'time_sec',
        'time_nanosec',
        'trigno_prompt',
        'trigno_delayed',
        'dt_last_ad_muon',
        'dt_last_ad_shower_muon',
        'dt_last_wp_muon'
        )
NPIXELS = 192
NCHANNELS = 4
NMETADATA = len(METADATA_NAMES)
ENTRYSIZE = NMETADATA + NCHANNELS * NPIXELS

def main():
    
    filelistname = 'yasufiles.txt' # ROOT files containing IBD candidates
    treename = 'tr_ibd'
    filelist = pandas.read_csv(filelistname, squeeze=True, header=None)
    Nstart = int( sys.argv[1]) 
    N = 10000 if len(sys.argv)== 2 else int(sys.argv[2])  # 10k events per file 
    start = Nstart * N
    stop = start + N 
    outfilename = 'ibd_yasu_%d_%d.h5' % (start, stop-1)
    data = np.zeros((N, ENTRYSIZE), dtype='float32')
    # find starting value
    global_index = 0
    for i, filename in enumerate(filelist):
        roottree = roottools.RootTree(filename, treename)
        numentries = roottree.numEntries()
        if global_index + numentries >= start:
            startfile = filename
            startfileindex = i
            # skip into the middle of the file to get to the correct entry
            startentry = start - global_index
            break
        else:
            global_index += numentries
    # Go through the starting file
    intbranches = ['runno', 'fileno', 'site', 'det', 'time_sec',
        'time_nanosec', 'trigno_prompt', 'trigno_delayed',
        'nHitsAD_prompt', 'nHitsAD_delayed']
    floatbranches = ['dt_last_ad_muon', 'dt_last_ad_shower_muon',
        'dt_last_wp_muon']
    ivectorbranches = ['hitCountAD_prompt', 'ring_prompt', 'column_prompt',
        'hitCountAD_delayed', 'ring_delayed', 'column_delayed']
    fvectorbranches = ['timeAD_prompt', 'chargeAD_prompt', 'timeAD_delayed',
        'chargeAD_delayed']
    roottree = roottools.RootTree(startfile, treename, intbranches,
        floatbranches, ivectorbranches, fvectorbranches)
    global_index = start  # refers to global index over all files
    endentry = min(roottree.numEntries(), startentry+N)
    for entrynum in range(startentry, endentry):
        roottree.loadentry(entrynum)
        event = roottree.current
        data[global_index-start] = getFlattenedData(event)
        global_index += 1
    remainingEntries = stop - global_index
    fileindex = startfileindex + 1
    while remainingEntries > 0 and fileindex < len(filelist):
        roottree = roottools.RootTree(filelist[startfileindex + 1],
            treename, intbranches, floatbranches,
            ivectorbranches, fvectorbranches)
        startentry = 0
        endentry = min(roottree.numEntries(), remainingEntries)
        for entrynum in range(startentry, endentry):
            roottree.loadentry(entrynum)
            event = roottree.current
            data[global_index-start] = getFlattenedData(event)
            global_index += 1
        remainingEntries = stop - global_index
    outfile = h5py.File(outfilename, 'w')
    # TODO determine if chunks/compression is necessary
    outdset = outfile.create_dataset("ibd_pair_data", data=data)
    # Set attributes so future generations can read this dataset
    outdset.attrs['description'] = \
"""This dataset contains pairs of IBD candidates (prompt, delayed) as
selected by the physics selection criteria.

See the 'structure' attribute for a detailed description of the dset layout.

Each row in the dataset contains flattened versions of the charge and time on
each PMT for both the prompt and delayed events, followed by some metadata.
There are currently %d metadata items, which can be accessed as attributes from
0 to %d. Reverse lookup is also implemented to get the index of a given piece
of metadata.""" % (NMETADATA, NMETADATA)
    outdset.attrs['structure'] = \
"""Data set structure: N rows by %d columns. Each column has the following:

0-%d: flattened 8x24 of prompt charge
%d-%d: flattened 8x24 of prompt time
%d-%d: flattened 8x24 of delayed charge
%d-%d: flattened 8x24 of delayed time
%d-%d: metadata such as trigger numbers, site and AD info, and dt between
triggers and to the last muon.""" % (ENTRYSIZE, NPIXELS-1, NPIXELS,
    2*NPIXELS-1, 2*NPIXELS, 3*NPIXELS-1, 3*NPIXELS, 4*NPIXELS-1, 4*NPIXELS,
    ENTRYSIZE-1)

    # Set metadata attributes so people know what's in the last bit of each row
    for i, name in enumerate(METADATA_NAMES):
        outdset.attrs[str(i)] = name
        outdset.attrs[name] = i

    outfile.close()
    return

def getFlattenedData(event):
    event['nHitsAD'] = event['nHitsAD_prompt']
    event['chargeAD'] = event['chargeAD_prompt']
    event['timeAD'] = event['timeAD_prompt']
    event['ring'] = event['ring_prompt']
    event['column'] = event['column_prompt']
    charge2d_prompt, time2d_prompt = roottools.getChargesTime(event)
    event['nHitsAD'] = event['nHitsAD_delayed']
    event['chargeAD'] = event['chargeAD_delayed']
    event['timeAD'] = event['timeAD_delayed']
    event['ring'] = event['ring_delayed']
    event['column'] = event['column_delayed']
    charge2d_delayed, time2d_delayed = roottools.getChargesTime(event)
    # Reshape everything into one long vector
    flatteneds = [
        # reshape(-1) produces a 1D array
        charge2d_prompt.reshape(-1),
        time2d_prompt.reshape(-1),
        charge2d_delayed.reshape(-1),
        time2d_delayed.reshape(-1),
        np.array([event[name] for name in METADATA_NAMES])
    ]
    return np.hstack(flatteneds)

def unflattenData(datavec):
    """Expect a 779-length 1D numpy array or similar. Split it into four 8x24
    images plus the metadata, and return a dict with the appropriate
    attributes.
    """
    event = {}
    shape = (8, 24)
    event['charge_prompt'] = datavec[0:NPIXELS].reshape(shape)
    event['time_prompt'] = datavec[NPIXELS:(2*NPIXELS)].reshape(shape)
    event['charge_delayed'] = datavec[(2*NPIXELS):(3*NPIXELS)].reshape(shape)
    event['time_delayed'] = datavec[(3*NPIXELS):(4*NPIXELS)].reshape(shape)
    for i, name in enumerate(METADATA_NAMES):
        event[name] = datavec[(4*NPIXELS+i)]
    return event


if __name__=='__main__':
    main()

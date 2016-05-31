# roottools by peter sadowski
import ROOT
import array
import numpy as np
import itertools

class RootTree():
    def __init__(self,filename, treename, intbranches=[], floatbranches=[],ivectorbranches=[],fvectorbranches=[]):
        ch = ROOT.TChain(treename)
        status = ch.Add(filename)
        #if status == 1:
        #    raise ValueError('Error: File %s does not have tree %s' % (filename, treename))
        branchPointers = {}
        branchDict = {}
        ch.SetMakeClass(1)
        for branchname in intbranches:
            branchPointers[branchname] = array.array('I', [0])
        for branchname in floatbranches:
            branchPointers[branchname] = array.array('f', [0])
        for branchname in fvectorbranches:
            branchPointers[branchname] = ROOT.std.vector('float')() 
        for branchname in ivectorbranches:
            branchPointers[branchname] = ROOT.std.vector('int')() 
       
        branches = intbranches + floatbranches + ivectorbranches + fvectorbranches
        ch.SetBranchStatus("*",0)
        [ ch.SetBranchStatus(branchname, 1) for branchname in branches ]
        for branchname in branches:
            branchDict[branchname] = ch.GetBranch(branchname)
            ch.SetBranchAddress(branchname, branchPointers[branchname])
        #return ch, branchDict, branchPointers, branches
        self.filename = filename
        self.treename = treename
        self.ch = ch
        self.branchDict = branchDict
        self.branchPointers = branchPointers
        self.branches = branches
        self.intbranches = intbranches
        self.floatbranches = floatbranches
        self.fvectorbranches = fvectorbranches
        self.ivectorbranches = ivectorbranches
        self.current = {} # Dict containing data for current entry.

    def loadentry(self, i):
        self.ch.LoadTree(i)
        self.ch.GetEntry(i)
        self.current = Entry(self, True)
        return self.current
     
    def getentries(self):
        ''' Return generator for all sequential entries.'''
        nEntries= self.ch.GetEntries()
        for i in xrange(nEntries):
            if i%1000==0:
                print "Processing event nr. %i of %i" % (i,nEntries)
            current = self.loadentry(i)
            yield current
    def numEntries(self):
        return self.ch.GetEntries()
        
    
    def find_trigger(self, detector, triggerNumber, startidx=0):
        ''' 
        Iterate over events quickly to find trigger.
        requirements = list of (branchname, value) pairs, eg. ('detector', 0)
        '''
        startidx = int(startidx)
        for i in xrange(startidx, self.ch.GetEntries()):
            self.ch.LoadTree(i)
            self.branchDict['detector'].GetEntry(i)
            if not self.branchPointers['detector'][0] == detector:
                continue
            self.branchDict['triggerNumber'].GetEntry(i) 
            if not self.branchPointers['triggerNumber'][0] == triggerNumber:
                continue
            return i
        raise Exception('Could not find d=%d tn=%d, biggest tn is %d' % (int(detector), int(triggerNumber),  self.branchPointers['triggerNumber'][0]))
        return None

class Entry(dict):
    '''This class stores the information in a TTree entry.

       NOTE: It constructs each numpy array separately in a lazy fashion so
       they are only built if needed.
    '''
    def __init__(self, parent_tree, lazy=True):
        super(Entry, self).__init__(self)
        self.parent = parent_tree
        if lazy:
            pass
        else:
            self.unlazyconstruct()

    def unlazyconstruct(self):
        parent = self.parent
        for branchname in parent.branches:
            self[branchname]

    def __getitem__(self, key):
        '''Fetch the item if it already exists, else construct it'''
        parent = self.parent
        if key in self:
            return dict.__getitem__(self, key)
        elif key in parent.intbranches:
            val = parent.branchPointers[key][0]
            dict.__setitem__(self, key, val)
            return val
        elif key in parent.floatbranches:
            val = parent.branchPointers[key][0]
            dict.__setitem__(self, key, val)
            return val
        elif key in parent.fvectorbranches:
            val_orig = parent.branchPointers[key]
            val = np.array(val_orig, dtype='float32')
            dict.__setitem__(self, key, val)
            return val
        elif key in parent.ivectorbranches:
            val_orig = parent.branchPointers[key]
            val = np.array(val_orig, dtype='int')
            dict.__setitem__(self, key, val)
            return val
        else:  # This will likely raise a KeyError
            return dict.__getitem__(self, key)


def getChargesTime(entry, preprocess_flag=True, dtype='float32'):
    ''' This function takes a readout entry and extracts the charge and time. '''
    charge = np.zeros((8, 24), dtype=dtype)
    time = np.zeros((8, 24), dtype=dtype)
    nHitsAD = entry['nHitsAD']
    chargeAD = entry['chargeAD']
    timeAD = entry['timeAD']
    ring = entry['ring'] - 1 # Convert to 0-idx
    column = entry['column'] - 1

    for hit in range(nHitsAD):
        if charge[ring[hit], column[hit]] != 0.0:
            # Second charge deposit in same PMT observed!
            time_orig = time[ring[hit], column[hit]]
            time_new = timeAD[hit]

            orig_in_window = (time_orig > -1650) and (time_orig < -1250)
            new_in_window = (time_new > -1650) and (time_new < -1250)
            if (new_in_window and not orig_in_window) or \
               (new_in_window and (time_new < time_orig)):
                # Use new
                pass
            else:
                continue
        charge[ring[hit], column[hit]] = chargeAD[hit]
        time[ring[hit], column[hit]] = timeAD[hit]
    if preprocess_flag:
        charge = preprocess(charge)
    return charge, time

def preprocess(X):
    ''' Preprocess charge image by taking log and dividing by scale factor.'''
    prelog = 1.0
    scale = 10.0 # log(500000) ~= 10
    X = np.maximum(X, np.zeros_like(X))
    X = np.log(X + prelog) / scale
    return X

def isflasher(entry):
    ''' Is this entry a flasher according to Yasu's cut.'''
    MaxQ = entry['MaxQ']
    Quadrant = entry['Quadrant']
    time_PSD = entry['time_PSD']
    time_PSD1 = entry['time_PSD1']
    MaxQ_2inchPMT = entry['MaxQ_2inchPMT']
    NominalCharge = entry['NominalCharge']
    eps = 10**-10
    flasher = not(\
              np.log10(Quadrant**2 + MaxQ**2/0.45/0.45 + eps) < 0.0 and \
              np.log10(4.0 * (1.0-time_PSD)**2 + 1.8 * (1.0-time_PSD1)**2 + eps) < 0.0 and \
              MaxQ_2inchPMT < 100.0) and (NominalCharge <= 3000.0)
    return flasher

def ismuon(entry):
    ''' Is this entry a muon according to Yasu's cut. '''
    NominalCharge = entry['NominalCharge']
    return NominalCharge > 3000.0    

def get_num_entries(filename):
    return get_num_readout_entries(filename)

def get_num_readout_entries(filename):
    t1 = makeCalibReadoutTree(filename)
    return t1.numEntries()

def get_num_stat_entries(filename):
    t2 = makeCalibStatsTree(filename)
    return t2.numEntries()

def makeCalibReadoutTree(filename):
    treename = '/Event/CalibReadout/CalibReadoutHeader'
    intbranches = ['nHitsAD','triggerNumber', 'detector']
    floatbranches = []
    ivectorbranches = ["ring","column","wallNumber"] #,"wallspot"]
    fvectorbranches = ["timeAD","chargeAD", "timePool", "chargePool", "wallSpot"]
    t1 = RootTree(filename, treename, intbranches=intbranches, floatbranches=floatbranches, ivectorbranches=ivectorbranches, fvectorbranches=fvectorbranches)
    return t1

def makeCalibStatsTree(filename):
    treename = '/Event/Data/CalibStats'
    floatbranches = ['MaxQ', 'Quadrant', 'time_PSD', 'time_PSD1', 'MaxQ_2inchPMT', 'NominalCharge'] #, 'dtLast_AD1_ms', 'dtLast_AD2_ms', 'dtLast_AD3_ms', 'dtLast_AD4_ms'] 
    intbranches = ['triggerNumber']#["detector","triggerNumber"] #,"triggerType","triggerTimeSec","triggerTimeNanoSec","nHitsAD","nHitsPool"]    
    ivectorbranches = []
    fvectorbranches = []
    t2 = RootTree(filename, treename, intbranches=intbranches, floatbranches=floatbranches, ivectorbranches=ivectorbranches, fvectorbranches=fvectorbranches)
    return t2
    

def rootfileiter(filename):
    t2 = makeCalibStatsTree(filename)
    t1 = makeCalibReadoutTree(filename)

    for entry1, entry2 in itertools.izip(t1.getentries(), t2.getentries()):
        entry1.update(entry2)
        yield entry1

def calibReadoutIter(filename):
    t1 = makeCalibReadoutTree(filename)
    for entry in t1.getentries():
        yield entry
        
    
def calibStatsIter(filename):
    t2 = makeCalibStatsTree(filename)
    for entry in t2.getentries():
        yield entry


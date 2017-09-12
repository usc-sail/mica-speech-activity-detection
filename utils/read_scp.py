### Derived and modified from kaldi_io by Rabbeh.
###
### Python modules to read from different custom types of
### .scp files produced from kaldi/post-processing kaldi
### outputs.
### 

###
### read_mat_scp : default reader for reading direct kaldi .scp output files
###                USAGE: key, mat = read_mat_scp(file_or_fd).next()
###
### read_mat_scp_spect : used for reading custom spectrogram .scp files appended
###                      with either single segment label, or frame level label,
###                      or no label at all.
###                      USAGE: key, mat, (lab) = read_mat_scp_spect(fl).next()
###
### read_mat_scp_seg_label : Reading .scp files, appended with single segment-level
###                          label, which is repeated to be same length as input
###                          data read
###                          USAGE: mat, labels = read_mat_scp_seg_label(fl).next()
###
### read_mat_scp_frame_label : Reading .scp files, appended with frame level-labels
###                            USAGE: mat, labels = read_mat_scp_frame_label(fl).next()
###

import numpy as np
import os, re, gzip, struct

def read_mat_scp(file_or_fd):
  """ generator(key,mat) = read_mat_scp(file_or_fd)
   Returns generator of (key,matrix) tuples, read according to kaldi scp.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Iterate the scp:
   for key,mat in kaldi_io.read_mat_scp(file):
     ...

   Read scp to a 'dictionary':
   d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
  """
  fd = open_or_fd(file_or_fd)
  try:
    for line in fd:
      (key,rxfile) = line.rstrip().split(' ')
      mat = read_mat(rxfile)
      yield key, mat
  finally:
    if fd is not file_or_fd : fd.close()

def read_mat_scp_spect(file_or_fd):
  """ generator(key,mat) = read_mat_scp(file_or_fd)
      Returns generator of (key, matrix,[label/s]) tuples, read according to kaldi scp.
      file_or_fd : scp, gzipped scp, pipe or opened file descriptor.
      
      Reads in input data from spectrogram .scp file, which may or may not be
      appended with frame/segment level labels.
  """
  fd = open_or_fd(file_or_fd)
  try:
    for line in fd:
      data = line.rstrip().split(' ')
      key = data[0]
      rxfile = data[1]
      mat = read_mat(rxfile)
      #mat_nodc = [x[1:] for x in mat]           ## Remove DC-element from spectrogram
      if len(data)==2:
          yield key, mat
      else:
          lab = [int(x) for x in data[2:]]
          yield key,mat, np.array(lab)
  finally:
    if fd is not file_or_fd : fd.close()

### Added by Rajat 07/12/17 for reading modified scp files:
def read_mat_scp_seg_label(file_or_fd):
  """ generator(mat,label) = read_mat_scp(file_or_fd)
   Returns generator of (matrix,labels) tuples, read according to kaldi scp.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Reads in input data and a single label from scp, and repeats it to length of 
   input data read from ark file.
  """
  fd = open_or_fd(file_or_fd)
  try:
    for line in fd:
      (key,rxfile,lab) = line.rstrip().split(' ')
      mat = read_mat(rxfile)
      labels = np.repeat(int(lab),len(mat))
      yield mat, labels
  finally:
    if fd is not file_or_fd : fd.close()

def read_mat_scp_frame_label(file_or_fd):
  """ generator(mat,labels) = read_mat_scp(file_or_fd)
   Returns generator of (matrix,labels) tuples, read according to kaldi scp.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Reads in input data, and labels equivalent to input length
   from ark file and returns them.
  """
  fd = open_or_fd(file_or_fd)
  try:
    for line in fd:
      data = line.rstrip().split(' ')
      key = data[0]
      rxfile = data[1]
      lab = [int(x) for x in data[2:]]
      labels = np.array(lab)
      mat = read_mat(rxfile)
      yield mat, labels
  finally:
    if fd is not file_or_fd : fd.close()

def read_mat(file_or_fd):
  """ [mat] = read_mat(file_or_fd)
   Reads single kaldi matrix, supports ascii and binary.
   file_or_fd : file, gzipped file, pipe or opened file descriptor.
  """
  fd = open_or_fd(file_or_fd)
  try:
    binary = fd.read(2)
    if binary == '\0B' : 
      mat = _read_mat_binary(fd)
    else:
      assert(binary == ' [')
      mat = _read_mat_ascii(fd)
  finally:
    if fd is not file_or_fd: fd.close()
  return mat

def _read_mat_binary(fd):
  # Data type
  type = fd.read(3)
  if type == 'FM ': sample_size = 4 # floats
  if type == 'DM ': sample_size = 8 # doubles
  assert(sample_size > 0)
  # Dimensions
  fd.read(1)
  rows = struct.unpack('<i', fd.read(4))[0]
  fd.read(1)
  cols = struct.unpack('<i', fd.read(4))[0]
  # Read whole matrix
  buf = fd.read(rows * cols * sample_size)
  if sample_size == 4 : vec = np.frombuffer(buf, dtype='float32') 
  elif sample_size == 8 : vec = np.frombuffer(buf, dtype='float64') 
  else : raise BadSampleSize
  mat = np.reshape(vec,(rows,cols))
  return mat

def _read_mat_ascii(fd):
  rows = []
  while 1:
    line = fd.readline()
    if (len(line) == 0) : raise BadInputFormat # eof, should not happen!
    if len(line.strip()) == 0 : continue # skip empty line
    arr = line.strip().split()
    if arr[-1] != ']':
      rows.append(np.array(arr,dtype='float32')) # not last line
    else: 
      rows.append(np.array(arr[:-1],dtype='float32')) # last line
      mat = np.vstack(rows)
      return mat

#################################################
# Data-type independent helper functions,

def open_or_fd(file, mode='rb'):
  """ fd = open_or_fd(file)
   Open file, gzipped file, pipe, or forward the file-descriptor.
   Eventually seeks in the 'file' argument contains ':offset' suffix.
  """
  offset = None
  try:
    # strip 'ark:' prefix from r{x,w}filename (optional),
    if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', file):
      (prefix,file) = file.split(':',1)
    # separate offset from filename (optional),
    if re.search(':[0-9]+$', file):
      (file,offset) = file.rsplit(':',1)
    # is it gzipped?
    if file.split('.')[-1] == 'gz':
      fd = gzip.open(file, mode)
    # input pipe?
    elif file[-1] == '|':
      fd = os.popen(file[:-1], 'rb')
    # output pipe?
    elif file[0] == '|':
      fd = os.popen(file[1:], 'wb')
    # a normal file...
    else:
      fd = open(file, mode)
  except TypeError: 
    # 'file' is opened file descriptor,
    fd = file
  # Eventually seek to offset,
  if offset != None: fd.seek(int(offset)) 
  return fd


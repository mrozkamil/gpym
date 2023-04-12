#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:53:08 2021

@author: km357
"""
#!/local/anaconda3/bin/Python3
# import sys
import subprocess
import os

home = os.path.expanduser("~")
gpm_dir = os.path.join(home, 'Data', 'GPM')
    
class PPS():
    def __init__(self, usr = None, save_dir = gpm_dir):
        if usr is None:
            print('Please provide a valid PPS user name')
            
        print('Data saved to "{}"'.format(save_dir) )
        self.server = 'https://arthurhouhttps.pps.eosdis.nasa.gov/text'
        self.dir_out = save_dir
        self.usr = usr
        
    def get_file_list(self,year, month, day, directory, prod, ver, gran_str):
        ''' Get the file listing for the given year/month/day
        using curl.
        Return list of files (could be empty).
        '''
        url = self.server + '/gpmallversions/V%s/' % ver + \
        '/'.join([year, month, day]) + \
        '/%s/' % directory
        cmd = 'curl --user %s:%s ' % (self.usr, self.usr) + url
        args = cmd.split()
        process = subprocess.Popen(args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
        stdout = process.communicate()[0].decode()
        if stdout[0] == '<':
            print ('No GPM files for the given date and product')
            return []
        file_list = stdout.split()
        file_list = [f for f in file_list if ((prod in f) and (gran_str in f))]
        return file_list
    
    def fetch_file(self,filename, verbose = False):
        ''' Get the given file from arthurhouhttps using curl. '''
        url = self.server + filename
        fname = filename.replace('/gpmallversions',self.dir_out)
        if os.path.exists(fname):
            if verbose:
                print('{} exists, skipped...'.format(fname))
               
        else:
            dirname = os.path.dirname(fname)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            
            cmd = 'curl --user %s:%s -o %s -z %s %s' % (
                self.usr, self.usr, fname, fname, url)
            
            args = cmd.split()
            if verbose:
                print('%s downloading...' % fname)
            process = subprocess.Popen(args,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            process.wait()  # wait so this program doesn't end
                            # before getting all files
            
            try:
                os.chmod(fname, 0o774)
            except:
                print('permissions cannot be changed')
            
    
    def download_files(self,date_str = '2015-01-01', directory = 'radar', 
                 prod = '', ver = 7, gran = None, verbose = False):
        """
        Parameters
        ----------
        date_str : str, optional
            Date string. The default is '2015-01-01'.
        directory : str, optional
            The directory name where the product file is stored.
            The default is 'radar'.
        prod : str, optional
            The GPM product name. The default is ''.
        ver : int, optional
            Product's version'
            The default is 6.
        gran : int, optional
            The orbit number. The default is None.
        verbose : Boolean, optional
            Determines if any messages are displayed. The default is False.

        Returns
        -------
        None.

        """
        year, month, day = date_str.split('-')
        if gran is None:
            gran_str = ''
        else:
            gran_str = '{:06d}'.format(gran)
            
        ver_str = '{:02d}'.format(ver)
        # loop through the file list and get each file
        file_list = self.get_file_list(year, month, day, directory, 
                                    prod, ver_str, gran_str, )
        for filename in file_list:
            # print(filename)
            if gran_str in filename:
                self.fetch_file(filename, verbose = verbose)
                
    def list_files(self,date_str = '2015-01-01', directory = 'radar', 
                 prod = '', ver = 7, gran = None, ):
        year, month, day = date_str.split('-')
        if gran is None:
            gran_str = ''
        else:
            gran_str = '{:06d}'.format(gran)
            
        ver_str = '{:02d}'.format(ver)
        # loop through the file list and get each file
        file_list = self.get_file_list(year, month, day, directory, 
                                    prod, ver_str, gran_str,)
        for filename in file_list:
            # print(filename)
            if gran_str in filename:
                print(filename)
# -*- coding: utf-8 -*-
"""
Kathryn Ozgun

Returns the json file containing the information for the requested title
"""

def get_from_OMBd(api_key, title):
    
    import ssl
    import urllib.request, urllib.parse, urllib.error
    import json
    
    # Service Oriented Approach
    # API - application program interface
    # data is shared between to companies
    # e.g. a standard and agreed upon way to share data/information
        
    if len(api_key)<1: api_key =  input('Enter API Key: ')
    
    
    # Ignore SSL certificate errors
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    
    serviceurl = 'http://www.omdbapi.com/?'
    
    if len(title)<1: title =  input('Enter title: ')
    
     
    parms = dict()
    if api_key is not False: parms['apikey'] = api_key
    parms['i'] = title
    url = serviceurl + urllib.parse.urlencode(parms)
    
    
    #print('Retrieving ', url)
    
    uh = urllib.request.urlopen(url, context=ctx)
    data = uh.read().decode()
    #print('Retrieved ', len(data), ' characters')
    
    
    try:
        js = json.loads(data)
    except:
        js = None
    
    #if not js or 'status' not in js or js['status']!='OK':
        #print('')
        #print(data)
    
    return js
    
    
#COnvert the MOA predicted files to JSON format
import simplejson as json
import re
import os
files=['em.pred','sea.pred','tj.pred']
path ='experimentation/MOA_AUE_NB/'

for filename in files:
    dataset = re.search('(.*).pred', filename, re.IGNORECASE)
    dataset= dataset.group(1)
    
    file_path=os.getcwd()+'/'+filename
    
    data=np.recfromcsv(file_path)
    error= [np.abs(data['real'][i]-data['pred'][i]) for i in range(0,len(data))]

    #find chunk accuracy
    total_accuracy=0
    chunk_size=1000

    chunk_accuracy_list= [1-np.mean(error[i:i+chunk_size]) for i in range(0,len(data),chunk_size)]
    overall_accuracy=1-np.mean(error)

    #write to JSON file
    op=[{"dataset":dataset,"chunk_size":chunk_size, "chunk_accuracy":chunk_accuracy_list,  "overall_accuracy":overall_accuracy, "error_list":error}]
    output_file=os.getcwd()+'/'+dataset+'.dat'
    f=open(output_file,'w')
    json.dump(op,f)
    f.close()
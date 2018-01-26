import os
import shutil
def modify_file(file_path,num_classes,ckpt_path,train_record_path,test_record_path,label_map_path):
    counter=0
    textbuf=''
    if os.name=='nt':
        ckpt_path=ckpt_path.replace('\\','\\\\')
        train_record_path=train_record_path.replace('\\','\\\\')
        test_record_path=test_record_path.replace('\\','\\\\')
        label_map_path=label_map_path.replace('\\','\\\\')
    with open(file_path,'r') as f:
        for line in f:
            txt=line
            stripped_text=txt.strip()
            if len(stripped_text)>0:
                if stripped_text[0]!='#':
                    #first we look for the num_classes thing
                    if stripped_text.startswith('num_classes'):
                        textbuf+='    num_classes: '+str(num_classes)+'\n'
                    elif stripped_text.startswith('train_input_reader'):
                        textbuf+=txt
                        counter+=1
                    elif stripped_text.startswith('eval_input_reader'):
                        textbuf+=txt
                        counter+=1
                    elif 'PATH_TO_BE_CONFIGURED' in stripped_text:
                        if stripped_text.startswith('fine_tune_checkpoint'):
                            textbuf+='  fine_tune_checkpoint: "'+ckpt_path+'"\n'
                        if counter==1:
                            if stripped_text.startswith('input_path'):
                                textbuf+='    input_path: "'+train_record_path+'"\n'
                            if stripped_text.startswith('label_map_path'):
                                textbuf+='  label_map_path: "'+label_map_path+'"\n'
                        if counter==2:
                            if stripped_text.startswith('input_path'):
                                textbuf+='    input_path: "'+test_record_path+'"\n'
                            if stripped_text.startswith('label_map_path'):
                                textbuf+='  label_map_path: "'+label_map_path+'"\n'

                    else:
                        textbuf+=txt
                else:
                    textbuf+=txt
            else:
                textbuf+=txt
    file_parent_folder=os.path.dirname(file_path)
    #create a backup copy
    shutil.copyfile(file_path,file_path+'.backup')
    with open(file_path,'w') as f:
        f.write(textbuf)
    #print(file,counter)

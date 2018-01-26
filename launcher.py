from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import os
import sys
import json
from random import randint
from time import sleep
from main_ui import Ui_frmmain
from time import gmtime, strftime
import xml.etree.ElementTree as ET
import subprocess
import shlex
from xml_to_csv import convert_xml_to_csv
import wget
import requests
import tarfile
import shutil
from config_modifier import modify_file
'''
Code for checking the settings file. It is really important for the GUI to know the path of tensorflow api folder path.
'''
SETTINGS_PATH=os.path.join(os.getcwd(),'settings.xml')
TFO_PATH=''
params_dict={}

class runcmd(QThread):
    output=pyqtSignal(str)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

    @pyqtSlot()
    def start(self): print("Thread started")

    def __del__(self):
        self.wait()

    @pyqtSlot(str)
    def run_command(self,command):
        process = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE, stderr=subprocess.STDOUT,shell=False)
        for line in process.stdout:
            print("out newline:")
            print(line)
        #print("done")
            self.output.emit(str(line,'utf-8'))
        #rc = process.poll()
        #return rc


class main_ui(QMainWindow):
    request_output=pyqtSignal(str)
    def __init__(self):
        global TFO_PATH
        super().__init__()

        self.ui = Ui_frmmain()
        self.ui.setupUi(self)
        self.load_tfo_path()
        self.ui.bttnbrowseprj.clicked.connect(self.load_prj_directory)
        self.ui.bttnbrowse.clicked.connect(self.load_dataset_directory)
        self.ui.bttngenerate.clicked.connect(self.generate_tfrecords_and_label_map)
        self.ui.bttnloadmodel.clicked.connect(self.load_selected_model)
        self.ui.bttneditconfig.clicked.connect(self.load_config_file)
        self.ui.bttntrain.clicked.connect(self.generate_train_script)
        self.ui.bttngettrainedmodel.clicked.connect(self.export_inference_graph)
        self.ui.bttnlaunchannotator.clicked.connect(self.launchannotator)
        self.load_model_list()

        self._thread=QThread()
        self._threaded=runcmd(output=self.get_output)
        self.request_output.connect(self._threaded.run_command)
        self._thread.started.connect(self._threaded.start)
        self._threaded.moveToThread(self._thread)
        qApp.aboutToQuit.connect(self._thread.quit)
        self._thread.start()

    def launchannotator(self):
        if os.path.exists(os.path.join(os.getcwd(),'labelImg'))==False:
            QMessageBox.warning(self,'Error','Please download the labelImg utility and paste the folder in tf_sensei directory.')
            return
        self.run_command('python3 labelImg/labelImg.py')

    def load_prj_directory(self):
        global params_dict
        folderpath=self.select_folder()
        if os.path.exists(folderpath)==False:
            return
        items=os.listdir(folderpath)
        self.ui.lblprojectpath.setText(folderpath)
        params_dict['project_path']=folderpath
        #now we do some checks to see if there is an existing project here.
        if os.path.exists(os.path.join(folderpath,'data')):
            if os.path.exists(os.path.join(folderpath,'data','train.record')):
                if os.path.exists(os.path.join(folderpath,'data','test.record')):
                    params_dict['data_path']=os.path.join(folderpath,'data')
                    self.ui.txtstatus.append('Existing project found. Loaded informations.')

            if os.path.exists(os.path.join(folderpath,'training')):
                if os.path.exists(os.path.join(folderpath,'training','training_conf.config')):
                    params_dict['config_file']=os.path.join(folderpath,'training','training_conf.config')
                    self.ui.txtstatus.append('Existing model config found. Loaded informations.')
                if os.path.exists(os.path.join(folderpath,'data','model_config.txt')):
                    with open(os.path.join(folderpath,'data','model_config.txt'),'r') as f:
                        params_dict['ckpt_path']=f.readline()
                else:
                    self.ui.txtstatus.append('No model info found. You need to load the model again')
        for keys,values in params_dict.items():
            print(keys,values)
    def load_dataset_directory(self):
        global params_dict
        projectpath=str(self.ui.lblprojectpath.text())
        print(projectpath)
        if os.path.exists(projectpath)==False:
            QMessageBox.warning(self,"Error","Invalid project path. Select a proper directory to continue.")
            return
        folderpath=self.select_folder()
        train_path=''
        test_path=''
        #now we run some tests, if this directory contains train and test folders,then those would be used
        #if not, it will check the contents of the directory and see if it has valid number of files(even numbered files)
        #with equal number of xml and image files,then we shall work on the xml files.
        if len(folderpath)==0:
            return
        if os.path.isdir(os.path.join(folderpath,'train')) and os.path.isdir(os.path.join(folderpath,'test')):
            train_path=os.path.join(folderpath,'train')
            test_path=os.path.join(folderpath,'test')
        else:
            QMessage.warning(self,"Error","Selected folder does not contain train and test directories. Create them and put annotated images there. You can open annotation tool from here and do the annotation and then split the files into train and test folders along with their annotation xml files and then select folder again.")
            return

        #now check train and test paths for possible errors(such as missing annotations etc.)
        train_files=os.listdir(train_path)
        test_files=os.listdir(test_path)
        if len(train_files)==0:
            QMessageBox.warning(self,"Error","No files in train directory.")
            return

        if len(test_files)==0:
            QMessageBox.warning(self,"Error","No files in test directory.")
            return


        #now we read the annotation files to get labels
        self.ui.txtstatus.append("Reading train folders to list labels...")
        labels=[]
        file_counter=0
        for file in train_files:
            if file.endswith('.xml'):
                file_counter+=1
                label_list=self.get_labels_from_file(os.path.join(train_path,file));
                for item in label_list:
                    if item not in labels:
                        labels.append(item)
        if len(item)==0:
            QMessageBox.warning(self,'Error','No annotation file found. Please launch the annotator tool to annotate the files and then reload the dataset directory.')
            return
        self.ui.lstlabels.addItems(labels)
        self.ui.txtstatus.append("Found "+str(len(labels))+" label(s) in "+str(file_counter)+" annotation file(s).")
        self.ui.lbldatafolder.setText(folderpath)
        params_dict['dataset_dir']=folderpath

    def generate_tfrecords_and_label_map(self):
        global params_dict
        #time to generate the labelmap first
        #self.run_command('df -h')
        self.ui.bttngenerate.setEnabled(False)
        self.ui.txtstatus.append('Generating Label map file...')
        project_path=str(self.ui.lblprojectpath.text())
        data_path=os.path.join(project_path,'data')
        if os.path.exists(data_path)==False:
            os.mkdir(data_path)
        file_str=''
        root = ET.Element("Label_map")
        for i in range(self.ui.lstlabels.count()):
            file_str=file_str+"item {"
            file_str=file_str+"\n  id: "+str((i+1))
            file_str=file_str+"\n  name: '"+str(self.ui.lstlabels.item(i).text())+"'"
            file_str=file_str+"\n}\n\n"
            element=ET.SubElement(root, "object")
            element.attrib['id']=str(i+1)
            element.text=str(self.ui.lstlabels.item(i).text())
        with open(os.path.join(data_path,'labels_trainer.pbtxt'),'w') as f:
            f.write(file_str)
        tree = ET.ElementTree(root)
        tree.write(os.path.join(data_path,'labels_trainer.xml'))
        self.ui.txtstatus.append('Done.')
        #we also created an xml file which would be useful for our tfrecord generator.
        #now we convert everything to csv
        dataset_path=str(self.ui.lbldatafolder.text())
        train_path=os.path.join(dataset_path,'train')
        test_path=os.path.join(dataset_path,'test')
        print(train_path)
        convert_xml_to_csv(train_path,test_path,data_path)
        self.ui.txtstatus.append('csv conversion done.')
        #now the tfrecord generation thing
        train_command=self.generate_cmd(data_path,dataset_path)
        test_command=self.generate_cmd(data_path,dataset_path,train=False)
        #print(train_command)
        self.run_command(train_command)
        self.run_command(test_command)
        print("Done")
        self.ui.bttngenerate.setEnabled(True)
        params_dict['data_path']=data_path

    def generate_cmd(self,data_folder,image_folder,train=True):
        command_to_run="python3 generate_tfrecord.py"
        if train==True:
            command_to_run+=" --csv_input="+os.path.join(data_folder,'train_labels.csv')
            command_to_run+=" --output_path="+os.path.join(data_folder,'train.record')
            command_to_run+=" --image_path="+os.path.join(image_folder,'train')
        else:
            command_to_run+=" --csv_input="+os.path.join(data_folder,'test_labels.csv')
            command_to_run+=" --output_path="+os.path.join(data_folder,'test.record')
            command_to_run+=" --image_path="+os.path.join(image_folder,'test')
        command_to_run+=" --label_map="+os.path.join(data_folder,'labels_trainer.xml')
        return command_to_run
    def get_labels_from_file(self,xmlfilepath):
        ret_list=[]
        tree = ET.ElementTree(file=xmlfilepath)
        for item in tree.iter():
            if item.tag=='name':
                ret_list.append(item.text)
        return list(set(ret_list))


    def check_even_image_xml(self,list_files):
        len_files=len(list_files)
        count_xml=0
        for file in list_files:
            if file.endswith('.xml'):
                count_xml+=1
        print(count_xml)
        if count_xml<len_files/2:
            return False
        return True

    @pyqtSlot(str)
    def run_command(self,command):
        self.request_output.emit(command)

    @pyqtSlot(str)
    def get_output(self,output):
        self.ui.txtstatus.append(output)


    def load_tfo_path(self):
        global TFO_PATH
        #performs the opearations needed for loading/updating the object detection api path
        if os.path.exists(SETTINGS_PATH)==False:
            QMessageBox.warning(self, "Warning", "No Settings file found. It seems this is a fresh start. So specify the path of the object_detection directory.")
            TFO_PATH=self.select_folder()
            self.create_settings()
        else:
            _tfo_path=self.get_tfo_path()
            if os.path.exists(_tfo_path)==False:
                QMessageBox.warning(self, "Error", "Tensorflow object detection api path not valid.please update it")
                self.update_tfo_path()
            else:
                TFO_PATH=_tfo_path
        self.ui.txtstatus.append('Tensorflow OD API path: '+TFO_PATH)

    def select_folder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        return folder

    def get_tfo_path(self):
        tree = ET.ElementTree(file=SETTINGS_PATH)
        root = tree.getroot()
        tfo_path=root.find('TFO_PATH')
        return tfo_path.text

    def update_tfo_path(self):
        global TFO_PATH
        folder=self.select_folder()
        tree = ET.parse(SETTINGS_PATH)
        root = tree.getroot()
        tfo_path=root.find('TFO_PATH')
        tfo_path.text=folder
        tree.write(SETTINGS_PATH)
        TFO_PATH=folder

    def create_settings(self):
        root = ET.Element("settings")
        ET.SubElement(root, "TFO_PATH").text = TFO_PATH
        tree = ET.ElementTree(root)
        tree.write(SETTINGS_PATH)

    def load_model_list(self):
        lst_names=[]
        print("OK")
        tree = ET.ElementTree(file='model_download.xml')
        root=tree.getroot()
        names=root.findall('.//name')
        print(names)
        for name in names:
            print(name.text)
            lst_names.append(name.text)
        self.ui.cbomodels.clear()
        self.ui.cbomodels.addItems(lst_names)


    def load_selected_model(self):
        global params_dict
        project_path=str(self.ui.lblprojectpath.text())
        if os.path.exists(project_path)==False:
            QMessageBox.warning(self,'Error','Invalid project path.')
            return
        selected_model=str(self.ui.cbomodels.currentText())
        tree = ET.ElementTree(file='model_download.xml')
        root=tree.getroot()
        models=root.findall('model')
        for model in models:
            if model[0].text==selected_model:
                url=model[1].text
                url_split=url.split('/')
                url_file_name=url_split[len(url_split)-1]
                download_path=os.path.join(TFO_PATH,url_file_name)
                if os.path.exists(download_path)==False:
                    QMessageBox.about(self,"Info","The model will now be downloaded to "+TFO_PATH+". Please do not delete the tar.gz file to avoid redownload. If you have a model already downloaded and modified somehow, please take its backup before pressing OK as it will be replaced.")
                    with open(download_path, "wb") as f:

                        self.ui.txtstatus.append("Downloading " + url_file_name)
                        response = requests.get(url, stream=True)
                        total_length = response.headers.get('content-length')

                        if total_length is None: # no content length header
                            f.write(response.content)
                        else:
                            dl = 0
                            total_length = int(total_length)
                            for data in response.iter_content(chunk_size=4096):
                                dl += len(data)
                                f.write(data)
                                done = int(100 * dl / total_length)
                                self.ui.dwp.setValue(done)
                                #sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )
                                #sys.stdout.flush()
                        #if we are here,the file has most likely been downloaded.
                if os.path.exists(download_path):
                    try:
                        tar = tarfile.open(download_path, 'r:gz')
                        model_folder_path=os.path.commonprefix(tar.getnames())
                        self.ui.txtstatus.append('Extracting...')
                        tar.extractall(path=TFO_PATH)
                        self.ui.txtstatus.append('Done.')
                        params_dict['ckpt_path']=os.path.join(TFO_PATH,model_folder_path,'model.ckpt')
                        if os.path.exists(os.path.join(project_path,'data'))==False:
                            os.mkdir(os.path.join(project_path,'data'))
                        params_dict['data_path']=os.path.join(project_path,'data')
                        with open(os.path.join(params_dict['data_path'],'model_config.txt'),'w') as f:
                            f.write(params_dict['ckpt_path'])
                    except:
                        QMessageBox.warning(self,"Error","Unknown Error")
                else:
                    QMessageBox.warning(self,"Error","Downloaded file not found.")
                return

    def load_config_file(self):
        global params_dict
        if os.path.exists(str(self.ui.lblprojectpath.text()))==False:
            QMessageBox.warning(self,"Error","Project path not selected. Operation can not continue.")
            return
        project_path=str(self.ui.lblprojectpath.text())
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select a config file", os.path.join(TFO_PATH,'samples','configs'),"config files (*.config);;All Files (*)", options=options)
        if fileName:
            training_path=os.path.join(project_path,'training')
            if os.path.exists(training_path)==False:
                os.mkdir(training_path)
            destination_file_path=os.path.join(training_path,'training_conf.config')
            shutil.copyfile(fileName,destination_file_path)
            self.ui.txtstatus.append('Config file copied to training directory')
            params_dict['config_file']=destination_file_path
            for keys,values in params_dict.items():
                print(keys,values)
            reply=QMessageBox.question(self,"Confirm Automatic modification","tf_sensei is capable of performing very basic modifications to the file(Adding number of classes, train and test tfrecord file path and label map file path).However,if you wish to make advanced changes like model configuration,batch_size etc, then you can change them by accessing the file at <project_path>/training directory. It's name would be training_conf.config. Now press Yes for auto modification and No for manual.", QMessageBox.Yes|QMessageBox.No)
            if reply==QMessageBox.Yes:

                modify_file(destination_file_path,int(self.ui.lstlabels.count()),params_dict['ckpt_path'],os.path.join(params_dict['data_path'],'train.record'),os.path.join(params_dict['data_path'],'test.record'),os.path.join(params_dict['data_path'],'labels_trainer.pbtxt'))
                self.ui.txtstatus.append("Done with modification. Please check for yourself to confirm.")

    #time for the grand finale
    def generate_train_script(self):
        global params_dict
        #Making sure everything was done
        pipeline_path=''
        training_dir=''
        try:
            prj=params_dict['project_path']
        except KeyError:
            QMessageBox.warning(self,'Error','No project found.')
            return
        try:
            pipeline_path=params_dict['config_file']
        except KeyError:
            QMessageBox.warning(self,'Error','No training directory found. Select a model and add a model config file to create it.')
            return

        training_dir=os.path.dirname(pipeline_path)

        train_command_str='#!/bin/bash\n'
        train_command_str+='python3 '+os.path.join(TFO_PATH,'train.py')
        train_command_str+=' --logtostderr --train_dir='+training_dir
        train_command_str+=' --pipeline_config_path='+pipeline_path
        with open(os.path.join(prj,'train_bash_file.sh'),'w') as f:
            f.write(train_command_str)
        os.system('chmod +x '+os.path.join(prj,'train_bash_file.sh'))
        QMessageBox.about(self,'Done',"Created the bash file for training. For now, this software won't run it automatically since there is an issue of early stopping and such. So please open a terminal in the project directory and run ./train_bash_file.sh to train your model.Also open another terminal in the same directory and run 'tensorboard --logdir training/' to start tensorboard. When you are done with training. Come back here and Click Export Inference graph button to create a prediction ready version fo the model.\nIf you are gettting OOM error from tensorflow, you can open <project_path>/training/training_conf.conf and reduce the batch size to see if it improves anything.")


    def export_inference_graph(self):
        params_dict
        #the final piece of the puzzle
        #we now do the export inference graph which would give us a usable model
        prj=''
        pipeline_path=''
        try:
            prj=params_dict['project_path']
        except KeyError:
            QMessageBox.warning(self,"Error","No project found.")
            return
        try:
            pipeline_path=params_dict['config_file']
        except KeyError:
            QMessageBox.warning(self,"Error","No model config file found")
            return

        training_dir=os.path.join(prj,'training')
        checkpoint_file_path=os.path.join(training_dir,'checkpoint')
        if os.path.exists(checkpoint_file_path)==False:
            QMessageBox.warning(self,'Error','No checkpoint file. Probably training was not performed or was deleted or corrupted.')
            return
        ckpt_file_path=''
        q=QMessageBox.question(self,'Question','TF sensei can detect the latest checkpoint automatically but you can also select one by yourself if you prefer. Pressing No will let you select the checkpoint file yourself. Pressing Yes will let TF sensei select it for you. How wou you like to proceed?',QMessageBox.Yes|QMessageBox.No)
        if q==QMessageBox.No:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            ckpt_file_path, _ = QFileDialog.getOpenFileName(self,"Select a ckpt file",training_dir,"All Files (*)", options=options)
            dir_ckpt=os.path.dirname(ckpt_file_path)
            ckpt_prefix=os.path.splitext(os.path.basename(ckpt_file_path))[0]
            ckpt_file_path=os.path.join(dir_ckpt,ckpt_prefix)
        else:
            with open(checkpoint_file_path,'r') as f:
                for line in f:
                    print(line)
                    strp_line=line.strip()
                    if len(strp_line)>0:
                        if strp_line[0]!='#':    #avoiding comments,if any
                            if strp_line.startswith('model_checkpoint_path:'):
                                #we found our line
                                ckpt_file_path_splt=strp_line.split(':')
                                ckpt_file_path=ckpt_file_path_splt[1]
                                ckpt_file_path=ckpt_file_path.strip()
                                ckpt_file_path=ckpt_file_path.replace('"','')
                                ckpt_file_path=ckpt_file_path.rstrip()
                                print(ckpt_file_path)
                                break

        #now,we have it all, so let's do it
        command_str='python3 '+os.path.join(TFO_PATH,'export_inference_graph.py')
        command_str+=' --input_type image_tensor'
        command_str+=' --pipeline_config_path '+pipeline_path
        command_str+=' --trained_checkpoint_prefix '+ckpt_file_path
        command_str+=' --output_directory '+os.path.join(prj,os.path.basename(prj)+'_output')
        print(command_str)
        self.run_command(command_str)
if __name__=="__main__":
    app = QApplication(sys.argv)
    form = main_ui()
    form.show()
    sys.exit(app.exec_())

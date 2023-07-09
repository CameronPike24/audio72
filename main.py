from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty, ObjectProperty
from kivy.clock import Clock
from jnius import autoclass
from audiostream import get_input
import wave
#
import os
from android.permissions import request_permissions,Permission,check_permission
from kivy_garden.graph import Graph, LinePlot, MeshLinePlot
import numpy as np 
from array import array
from datetime import datetime
 
#import tensorflow as tf
import zipfile
import wave, struct
import audioop
import sys


try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
  from tflite_runtime.interpreter import load_delegate
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf
  Interpreter = tf.lite.Interpreter
  load_delegate = tf.lite.experimental.load_delegate

 
 
#if not os.path.isdir("/sdcard/kivyrecords/"):
#    os.mkdir("/sdcard/kivyrecords/")

PATH = "rec_test1.wav"
 
recordtime = 5
samples_per_second = 60

 
 
class RootScreen(BoxLayout): #
    pass
 


       

class Recorder(object):
    def __init__(self):
        # get the needed Java classes
        self.MediaRecorder = autoclass('android.media.MediaRecorder')
        self.AudioSource = autoclass('android.media.MediaRecorder$AudioSource')
        self.AudioFormat = autoclass('android.media.AudioFormat')
        self.AudioRecord = autoclass('android.media.AudioRecord')
        # define our system
        self.SampleRate = 44100
        #self.SampleRate = 16000
        self.ChannelConfig = self.AudioFormat.CHANNEL_IN_MONO
        self.AudioEncoding = self.AudioFormat.ENCODING_PCM_16BIT
        self.BufferSize = self.AudioRecord.getMinBufferSize(self.SampleRate, self.ChannelConfig, self.AudioEncoding)
        #self.outstream = self.FileOutputStream(PATH)
        self.sData = []
        #self.mic = get_input(callback=self.mic_callback, source='mic', buffersize=self.BufferSize)
        self.mic = get_input(callback=self.mic_callback, source='default', buffersize=self.BufferSize)
        print("This is the audio source")
        print(self.AudioSource)
        print("This is the mic channels")
        print(self.mic.channels)
        print("self.BufferSize")
        print(self.BufferSize)
        
        
        
        ###################################################
        # Step 1 - Load bird model
        ######################################################

        # Load TFLite model and allocate tensors.
        #This is to load the default yamnet model which you dont need to do
        #with open('lite-model_yamnet_tflite_1.tflite', 'rb') as fid:

        #Open the bird model
        with open('my_birds_model.tflite', 'rb') as fid:
            tflite_model = fid.read()    

        #interpreter = tf.lite.Interpreter('lite-model_yamnet_tflite_1.tflite')
        #interpreter = tf.lite.Interpreter(model_content=tflite_model)             
        self.interpreter = Interpreter(model_content=tflite_model) 
        
        
        ###################################################
        # Step 2 - Get labels/bird names from metadata inside mdel
        ######################################################

        try:
            #with zipfile.ZipFile('my_birds_model.tflite') as model_with_metadata:
            #with zipfile.ZipFile('./birds_models/my_birds_model.tflite') as model_with_metadata: 
            with zipfile.ZipFile('my_birds_model.tflite') as model_with_metadata: 
  
                if not model_with_metadata.namelist():
                    raise ValueError('Invalid TFLite model: no label file found.')


                #the file name below retrieves the default Yamnet model metadata
                #file_name = model_with_metadata.namelist()[0]
                #or
                #the file name below retrieves the new bird model metadata namely [1]
                file_name = model_with_metadata.namelist()[1]

                with model_with_metadata.open(file_name) as label_file:
                    label_list = label_file.read().splitlines()
                    print("label_list")
                    print(label_list)
                    #_label_list = [label.decode('ascii') for label in label_list]
                    self._label_list = [label.decode('utf-8') for label in label_list]
      
                    print("_label_list")
                    print(self._label_list)  #Should print list of bird codes ['azaspi1', 'chcant2', 'houspa', 'redcro', 'wbwwre1']
      
                    print(len(self._label_list))  # should print 5   
      
                    #print(test_data.index_to_label[0])
      
        except zipfile.BadZipFile:
            print('ERROR: Please use models trained with Model Maker or downloaded from TensorFlow Hub.')
            raise ValueError('Invalid TFLite model: no metadata found.')        
        
      
        
        ###################################################
        # Step 3 - Get input/output details of model to ensure the audio sample fits the model input
        ######################################################

        self.input_details = self.interpreter.get_input_details()
        print("input_details")
        print(self.input_details)
        self.waveform_input_index = self.input_details[0]['index']
        self.output_details = self.interpreter.get_output_details()
        print("output_details")
        print(self.output_details)

        #Dont need this it is for the default Yamnet model
        self.scores_output_index = self.output_details[0]['index']
        print("scores_output_index 0")
        print(self.scores_output_index)

        #Get the output scores of the bird model
        self.scores_output_index1 = self.output_details[1]['index']
        print("")
        #Shows 116
        print(self.scores_output_index1)

        #The below is to create a wavfile of silence for testing only
        #Input: 0.975 seconds of silence as mono 16 kHz waveform samples.
        #waveform = np.zeros(int(round(0.975 * 16000)), dtype=np.float32)        
        
 
    def mic_callback(self, buf):
        self.sData.append(buf)
        print ('got : ' + str(len(buf)))
        print(self.sData)
        
        # convert our byte buffer into signed short array
        '''
        data_struct = struct.unpack("<h",buf)
        sound[0] = int(data_struct[0])
        print("sound[0]")
        print(sound[0])
        '''
        
        # assuming myByte is your byte array:
        len_count = len(buf)/2

        #sh = struct.unpack('h'*len_count , buf)
        #sh = np.array(struct.unpack('<h', buf))
        #print("sh")
        #print(sh)
        
        graph_values = np.frombuffer(buf, np.int16)
        print("graph_values")
        print(graph_values)
        
        
        #values = array("h", buf)
    

        # get right values only
        #r_values = values[1::2] # gives you every second value eg [1,2,3,4,5,6] would give [2,4,6,8]
        #r_values = values[1::1]
        
        
        #r_values = values[:] #a copy of all elements
        #print("r_values")
        #print(r_values)
        
        
        
        #Plot graph
        #graph_values = r_values[:]
        
        #graph_values = np.array(struct.unpack('<h', graph_values))
        #graph_values = np.array(graph_values ) / 32767
        #print("graph_values")
        #print(graph_values)
        
        
        
        '''
        self.record_form = RecordForm()
        self.record_form.get_value(graph_values)
        '''
        
        
        
        '''
        # reduce by 20%
        r_values = map(lambda x: x * 0.8, r_values)
        r_values = int(r_values)
        print("r_values")
        print(r_values)
        
        # you can assign only array for slice, not list
        # so we need to convert back list to array
        values[:] = array("h", r_values)
        print("values[:] ")
        print(values)
        
        # convert back the array to a byte buffer for speaker
        #sample.write(values.tostring())
        final = values.tostring()
        print("final string value")
        print(final)
        '''
 
    def start(self):
        self.mic.start()
        Clock.schedule_interval(self.readbuffer, 1/samples_per_second)
        print("def start(self)")
        a = datetime.now().strftime('%d-%m-%Y %H:%M:%S') 
        print(a)
 
    def readbuffer(self, dt):
        self.mic.poll()
 
    def dummy(self, dt):
        print ("dummy")
 
    def stop(self):
        Clock.schedule_once(self.dummy, 0.5)
        Clock.unschedule(self.readbuffer)
        self.mic.stop()
        wf = wave.open(PATH, 'wb')
        wf.setnchannels(self.mic.channels)
        wf.setsampwidth(2)
        wf.setframerate(self.mic.rate)
        wf.writeframes(b''.join(self.sData))
        print("we at stop")
        wf.close()
        
        print("files in pwd")
        print(os.listdir())
        
        print("contents of sData")
        print(self.sData)
        
        print("finished creating wav file")     
        b = datetime.now().strftime('%d-%m-%Y %H:%M:%S') 
        print(b)
        #No need to play the file
        #self.play()                    
        self.audio_path  = "rec_test1.wav"   
             
        #Downsample the audio from the microphones 44100 down to 16000 that the model was trained on
        self.audio_path_out  = "rec_test2.wav"
        self.audio_inrate = 44100
        self.audio_outrate = 16000
        self.audio_inchannels = 1
        self.audio_outchannels = 1      
        
        #Start downsample 
        self.downsample_success = self.downsampleWav(self.audio_path, self.audio_path_out, self.audio_inrate, self.audio_outrate, self.audio_inchannels, self.audio_outchannels)        
                
        if self.downsample_success == True:
            g = datetime.now().strftime('%d-%m-%Y %H:%M:%S')       
            print("downsampling completed at this time")
            print(g)         
            self.prepare_audio_frames()
        else:
            print("Could not downsample")          
        


    '''        
    def play(self):
        MediaPlayer = autoclass('android.media.MediaPlayer')
        AudioManager = autoclass('android.media.AudioManager')

        self.sound = MediaPlayer()
        #self.sound.setDataSource(yourDataSource) #you can provide any data source, if its on the devie then the file path, or its url if you are playing online
        #self.sound.setDataSource('testaudio.mp4') 
        #self.audio_path = self.storage_path + "/wav/output2.wav"
        #self.audio_path = self.storage_path + "/wav/output1.wav" ##cant find folder
     
        
        
        
        #self.audio_path = dirCheck1 + "/wav/output1.wav"
        #self.audio_path = "/storage/emulated/0/org.example.c4k_tflite_audio1/wav/output1.wav"##Not found
        #self.audio_path = "/data/data/org.example.c4k_tflite_audio1/files/app/wav/output2.wav"
        #self.audio_path = "/data/data/org.example.c4k_tflite_audio1/files/app/output2.wav"        
        #self.audio_path  = "/data/data/org.example.c4k_tflite_audio1/files/app/testaudio.mp4"
        self.audio_path  = "rec_test1.wav"
     
        self.sound.setDataSource(self.audio_path) 
        self.sound.prepare()
        self.sound.setLooping(False) #you can set it to true if you want to loop     
        
        
     
        print("start play")
        e = datetime.now().strftime('%d-%m-%Y %H:%M:%S')       
        print("time of start playing audio")
        print(e)
        self.sound.start()
        
        # You can also use the following according to your needs
        #self.sound.pause()
        #self.sound.stop()
        #self.sound.release()
        #self.sound.getCurrentPosition()
        #self.sound.getDuration()   
        
        #self.downsample()
        #Downsample the audio from the microphones 44100 down to 16000 that the model was trained on
        self.audio_path_out  = "rec_test2.wav"
        self.audio_inrate = 44100
        self.audio_outrate = 16000
        self.audio_inchannels = 1
        self.audio_outchannels = 1
        
        print("end of play")
        f = datetime.now().strftime('%d-%m-%Y %H:%M:%S')       
        print("end time of playing audio")
        print(f)        
        
    '''    
   
      
      
      
    def downsampleWav(self, src, dst, inrate, outrate, inchannels, outchannels):
        '''
        if not os.path.exists(src):
            print('Source not found!')
            return False

        if not os.path.exists(os.path.dirname(dst)):
            os.makedirs(os.path.dirname(dst))
        '''
        
        h = datetime.now().strftime('%d-%m-%Y %H:%M:%S')       
        print("downsampling started at this time")
        print(h)  
        
        try:
            s_read = wave.open(src, 'r')
            s_write = wave.open(dst, 'w')
        except:
            print('Failed to open files!')
            return False

        n_frames = s_read.getnframes()
        data = s_read.readframes(n_frames)

        try:
            converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
            if outchannels == 1 & inchannels != 1:
                converted[0] = audioop.tomono(converted[0], 2, 1, 0)
        except:
            print('Failed to downsample wav')
            return False

        try:
            s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
            s_write.writeframes(converted[0])
        except:
            print('Failed to write wav')
            return False

        try:
            s_read.close()
            s_write.close()
        except:
            print('Failed to close wav files')
            return False

        return True      
      
       
      
        
 
  
    def prepare_audio_frames(self):
    
        ###################################################
        #Step 4 - Prepare wav file to be tested . Test different examples. When live on android use the captured sound to test with so you dont need 
        ######################################################
        
        self.random_audio = self.audio_path_out

        #self.random_audio = 'House Sparrow.wav'
        #self.random_audio = 'White-breasted Wood-Wren.wav'
        #self.random_audio = 'Azara_s Spinetail.wav'
        #self.random_audio = 'Chestnut-crowned Antpitta.wav'
        #self.random_audio = 'Red Crossbill.wav'  
        #self.random_audio = 'House_Sparrow_downsampled.wav' 
 
 
        self.wav_file = wave.open(self.random_audio, 'rb')
        # from .wav file to binary data in hexadecimal
        self.binary_data = self.wav_file.readframes(self.wav_file.getnframes())
        # from binary file to samples
        self.audio_data = np.array(struct.unpack('{n}h'.format(n=self.wav_file.getnframes()*self.wav_file.getnchannels()), self.binary_data))
        #print("output from wave")
        #print(audio_data)
        #print(len(audio_data))

        print("audio before tf.int16.max")
        print(self.audio_data)


        #tf.int16.max prints out as 32767 so just use this instead
        #audio_data = np.array(audio_data) / tf.int16.max

        self.audio_data = np.array(self.audio_data) / 32767

        #input_size = serving_model.input_shape[1]

        ###This is the size of the frame. 0.975 secomds long and is 16000 samples = 15600. try changing but you need
        #to train the model on the different rate as well
        #Dont need input size not using tf
        #input_size = 15600
        #print("input size")
        #print(input_size)
        print("audio before tf.signal.frame")
        print(self.audio_data)


        #def framing(sig, fs=16000, win_len=0.025, win_hop=0.01):
        #def framing(sig, fs=15600, win_len=0.025, win_hop=0.01):
        #def framing(sig, fs=15600, win_len=1, win_hop=0.975):
        #def framing(sig, fs=15600, win_len=1, win_hop=0.01):  ####runs forever
        def framing(sig, fs=15600, win_len=1, win_hop=1):   #####working one
            """
            transform a signal into a series of overlapping frames.

            Args:
            sig            (array) : a mono audio signal (Nx1) from which to compute features.
            fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
            win_len        (float) : window length in sec.
                                 Default is 0.025.
            win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.

            Returns:
            array of frames.
            frame length.
            """
            # compute frame length and frame step (convert from seconds to samples)
            frame_length = win_len * fs
            frame_step = win_hop * fs
            signal_length = len(sig)
            frames_overlap = frame_length - frame_step

            # Make sure that we have at least 1 frame+
            num_frames = np.abs(signal_length - frames_overlap) // np.abs(frame_length - frames_overlap)
            rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)

            # Pad Signal to make sure that all frames have equal number of samples
            # without truncating any samples from the original signal
            if rest_samples != 0:
                pad_signal_length = int(frame_step - rest_samples)
                z = np.zeros((pad_signal_length))
                pad_signal = np.append(sig, z)
                num_frames += 1
            else:
                pad_signal = sig

            # make sure to use integers as indices
            frame_length = int(frame_length)
            frame_step = int(frame_step)
            num_frames = int(num_frames)

            # compute indices
            idx1 = np.tile(np.arange(0, frame_length), (num_frames, 1))
            idx2 = np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
            indices = idx1 + idx2
            self.frames = pad_signal[indices.astype(np.int32, copy=False)]
            return self.frames



        #Split the received audio into segments of 15600 frames and pad the end with 0's
        #splitted_audio_data = tf.signal.frame(audio_data, input_size, input_size, pad_end=True, pad_value=0)

        #Use the numpy version not tensorflow
        self.splitted_audio_data = framing(self.audio_data)


        #waveform = splitted_audio_data[0]
        #waveform = np.float32(waveform)
        self.waveform = self.splitted_audio_data
        self.waveform = np.float32(self.waveform)
        print("waveform data")
        print(self.waveform)

        #This will show how many rows of frames there are eg (28, 15600) is 28 rows of frames
        print("waveform.shape")
        print(self.waveform.shape) 
 
        self.start_inference()
 





 
 
    def start_inference(self):    
       
        ######################################################################
        #Step 6 - Start inference
        #####################################################################
        results_array = []
        for i, data in enumerate(self.waveform):
            print("data in enumerate")
            print(data)
            #waveform = np.reshape(waveform, (1, 15600))
            inference_results = self.perform_inference(self.waveform_input_index,data)
            results_array.append(inference_results)
            print("results in enumerate at inference")
            print(results_array)
  
  


        results_np_array = np.array(results_array)
        print("results_np_array")
        print(results_np_array)
        mean_results_array = results_np_array.mean(axis=0)
        print("mean_results_array at inference")
        print(mean_results_array)
        result_index_array = mean_results_array.argmax()
        print("result_index_array")
        print(result_index_array)
        #print(f'Mean result: {test_data.index_to_label[result_index]} -> {mean_results[result_index]}')
        #print(_label_list_1[top_class_index]) 
        print("bird class")
        print(self._label_list[result_index_array])  
        
        j = datetime.now().strftime('%d-%m-%Y %H:%M:%S')       
        print("End of prediction")
        print(j)  
 



    def perform_inference(self,wave_input_index,audio_frame):

        ######################################################################
        #Step 5 - loop through audio frames and perform inference
        #####################################################################

        #setup as a method 
        audio_frame = np.reshape(audio_frame, (1, 15600))
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(wave_input_index, audio_frame)
        print("waveform_input_index")
        print(wave_input_index)
        self.interpreter.invoke()
        #The below one is to use the default model - dont use for now
        #scores = interpreter.get_tensor(scores_output_index)
        #scores = interpreter.get_tensor(test_data.index_to_label)
        #scores = interpreter.get_tensor(index_to_label)
        #scores = interpreter.get_tensor(test_data)
        #Below is the correct one to use for the bird model
        self.scores = self.interpreter.get_tensor(self.scores_output_index1)
    
        top_class_index = self.scores.argmax()
        print("top_class_index")
        print(top_class_index)

        print(self._label_list[top_class_index])  # Should print code for bird.
        #print(_label_list[spec_result_index])  # Should print name for bird.
        #print("scores")
        #print(scores)
        #print(scores.shape)  # Should print (1, 5)            
           
        return self.scores


           
 
REC = Recorder()
'''
class RecordApp(App):
	
    def __init__(self, **kwargs):
        super(RecordApp, self).__init__(**kwargs)
        
        	
 
    def build(self):
        #request_permissions([Permission.INTERNET, Permission.RECORD_AUDIO,Permission.READ_EXTERNAL_STORAGE,Permission.WRITE_EXTERNAL_STORAGE])
        self.title = 'Recording Application'
        return RecordForm()
        #return Builder.load_file("look.kv")
  
'''     
        
class RecordForm(BoxLayout): #
    #b_record = ObjectProperty()
    #p_bar = ObjectProperty()
    def __init__(self, **kwargs):
        super(RecordForm, self).__init__(**kwargs)    
        self.plot = MeshLinePlot(color=[1, 0, 0, 1])
 
    def start_record(self):
        #self.b_record.disabled = True
        #self.p_bar.max = recordtime
        #REC.prepare()
        REC.start()
        self.ids.graph.add_plot(self.plot)
        Clock.schedule_interval(self.get_value, 0.1)
        #Clock.schedule_interval(self.get_value, 0.1)
        #lock.schedule_interval(self.get_value, 1/samples_per_second)
        
        #stop recording after recordtime value
        Clock.schedule_once(self.stop_record, recordtime)
        #Clock.schedule_interval(self.update_display, 1/30.)
 
    def stop_record(self, dt):
        #Clock.unschedule(self.update_display)
        #self.p_bar.value = 0
        REC.stop()
        Clock.unschedule(self.get_value)
        #lock.unschedule(self.get_value)
        #self.b_record.disabled = False
        
    #def get_value(self, dt):
    #def get_value(self, array_values):
    def get_value(self, dt):
    
        #print("r_values for graph")
        #print(array_values)
        array_values = [100,150,200,25,200,230,130,170,200,300,399,200,50,45,300,34,33,49,230,120,156,200,40,45,100,200,120,95,39,44,77,99,30]
        #NB j//5 not j/5 removes any remainder which results in a float
        self.plot.points = [(i, j//5) for i, j in enumerate(array_values)] 
        #self.plot.points = [(i, j//5) for i, j in enumerate(islice(array_values, 300))] 
        
        print("self.plot.points")
        print(self.plot.points)
        #self.plot.points = [(i, j/100) for i, j in enumerate(r_values)]         
        
 
    def update_display(self,dt):
        #self.p_bar.value = self.p_bar.value + dt
        print("here")        
        
'''
class JKMain(AnchorLayout):
    def __init__(self, **kwargs):
        super(JKMain, self).__init__(**kwargs)

    def change_text(self, layers):
        self.the_time.text = "Total Layers : " + str(layers)
        print("Total Layers = " + str(layers))

    def popup_func(self):

        content = ConfirmPopup()
        content.bind(on_answer=self._on_answer)
        self.popup = Popup(title="Select .zip file",
                           content=content,
                           size_hint=(None, None),
                           size=(500, 500),
                           auto_dismiss=False)
        self.popup.open()

    def _on_answer(self, instance, answer, obj):
        self.popup.dismiss()
'''

class Main(App):

    def build(self):
        #return JKMain()
        request_permissions([Permission.INTERNET, Permission.RECORD_AUDIO,Permission.READ_EXTERNAL_STORAGE,Permission.WRITE_EXTERNAL_STORAGE])
        return RecordForm()
        


if __name__ == "__main__":
    Main().run()



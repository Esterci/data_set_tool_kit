import pickle

class ProgBar:

    def __init__(self, n_elements,int_str):
        
        import sys

        self.n_elements = n_elements
        self.progress = 0

        print(int_str)

        # initiallizing progress bar

        info = '{:.2f}% - {:d} of {:d}'.format(0,0,n_elements)

        formated_bar = ' '*int(50)

        sys.stdout.write("\r")

        sys.stdout.write('[%s] %s' % (formated_bar,info))

        sys.stdout.flush()

    def update(self,prog_info=None):
        
        import sys

        if prog_info == None:

            self.progress += 1

            percent = (self.progress)/self.n_elements * 100 / 2

            info = '{:.2f}% - {:d} of {:d}'.format(percent*2,self.progress,self.n_elements)

            formated_bar = '-'* int (percent) + ' '*int(50-percent)

            sys.stdout.write("\r")

            sys.stdout.write('[%s] %s' % (formated_bar,info))

            sys.stdout.flush()


        else:

            self.progress += 1

            percent = (self.progress)/self.n_elements * 100 / 2

            info = '{:.2f}% - {:d} of {:d} '.format(percent*2,self.progress,self.n_elements) + prog_info

            formated_bar = '-'* int (percent) + ' '*int(50-percent)

            sys.stdout.write("\r")

            sys.stdout.write('[%s] %s' % (formated_bar,info))

            sys.stdout.flush()

class unite_files:

    def __init__(self,input_folder,output_folder='separeted_db/'):

        import glob

        self.out_folder = output_folder
        self.folder = input_folder
        self.file_list = glob.glob(input_folder+'*')
        self.file_list.sort()

    def convert(self):

        import pickle
        import numpy as np

        bar = ProgBar(len(self.file_list),"Reading and appending files...")

        for i,file in enumerate(self.file_list):

            data = pickle.load(open(file, "rb"))

            aux = [data[0,0],1,data[0,2],data[0,3],data[0,4],data[0,5],data[0,6],data[0,7]]

            data = np.vstack((aux,data))

            if i < 12:

                if i == 0:
                    adequate = data

                else:
                    adequate = np.vstack((adequate,data))

            elif i >= 12 and i < 14:

                if i == 12:
                    
                    pickle.dump(adequate,
                    open(self.out_folder+"adequate_tools.pkl", "wb"), 
                    )

                    del adequate

                    intermediate = data


                else:
                    intermediate = np.vstack((intermediate,data))

            elif i >= 14:

                if i == 14:
                    
                    pickle.dump(intermediate,
                    open(self.out_folder+"separeted_db/intermediate_tools.pkl", "wb"), 
                    )

                    del intermediate
                    
                    inadequate = data

                else:
                    inadequate = np.vstack((inadequate,data))

            bar.update()

        pickle.dump(inadequate,
                open(self.out_folder+"separeted_db/inadequate_tools.pkl", "wb"), 
                )

        del inadequate

class db_reduction:

    def __init__(self,input_folder,data_output_folder='iteration_data/', target_output_folder='iteration_target/',n_iterations = 33,maintained_percent=0.3, timeseries_reduction=0.4):
        
        import glob
        import os

        self.file_list = glob.glob(input_folder+'*')
        self.file_list.sort()

        self.input_folder = input_folder

        if not os.path.exists(data_output_folder):
            print(f"Creating '{data_output_folder}' folder")
            os.mkdir(data_output_folder)
        self.data_output_folder = data_output_folder
    
        if not os.path.exists(target_output_folder):
            print(f"Creating '{target_output_folder}' folder")
            os.mkdir(target_output_folder)
        
        self.target_output_folder = target_output_folder
        self.maintained_percent = maintained_percent
        self.n_iterations = n_iterations

        self.timeseries_reduction = timeseries_reduction

    def transform(self):

        import numpy as np
        from sklearn.model_selection import train_test_split

        bar = ProgBar(len(self.file_list*self.n_iterations),"Reducing number of time series...")

        for i in range(self.n_iterations):
            for j,file in enumerate(self.file_list):

                data = pickle.load(open(file, "rb"))

                n_measures = int(np.max(data[:, 1]))
                n_ids = len(np.unique(data[:,0]))

                expanded_dimensions = data.reshape(n_ids, n_measures, 8)

                new_n_measures = int(self.timeseries_reduction * n_measures)
                expanded_dimensions = expanded_dimensions[:, :new_n_measures, :]

                aux,reduced_data = train_test_split(expanded_dimensions,test_size=self.maintained_percent,
                                                    shuffle=False)
                
                n_ids = len(np.unique(reduced_data[:,0]))
                
                reduced_data = reduced_data.reshape(len(reduced_data)*new_n_measures, 8)

                del data, expanded_dimensions
                del aux

                if j < 12:

                    if j == 0:

                        it_data = reduced_data

                        it_target = np.zeros(len(reduced_data))

                    else:

                        it_data = np.vstack((it_data,reduced_data))

                        target = np.zeros(len(reduced_data))
                        
                        it_target = np.hstack((it_target,target))


                elif j >= 12 and j < 14:

                    it_data = np.vstack((it_data,reduced_data))

                    target = np.ones(len(reduced_data))
                    
                    it_target = np.hstack((it_target,target))

                elif j >= 14:

                    it_data = np.vstack((it_data,reduced_data))

                    target = np.ones(len(reduced_data)) + 1
                    
                    it_target = np.hstack((it_target,target))

                bar.update()

            pickle.dump(it_data,
                        open(self.data_output_folder + "data_mainteined_percent__{}__it__{}.pkl".format(
                            self.maintained_percent,i
                            ), "wb"), 
                            )

            pickle.dump(it_target,
                        open(self.target_output_folder + "target_mainteined_percent__{}__it__{}.pkl".format(
                            self.maintained_percent,i
                            ), "wb"), 
                            )

class sliding_window():
    def __init__ (self, data_input_folder, target_input_folder, data_output_folder='sliding_window_data/', target_output_folder='sliding_window_target/', step=100, size=1000):
        import glob
        import os

        self.step = step
        self.size = size

        self.file_list = glob.glob(data_input_folder+'*')
        self.file_list.sort()

        self.target_list = glob.glob(target_input_folder+'*')
        self.target_list.sort()
        
        if not os.path.exists(data_output_folder):
            print(f"Creating '{data_output_folder}' folder")
            os.mkdir(data_output_folder)
        self.data_output_folder = data_output_folder
    
        if not os.path.exists(target_output_folder):
            print(f"Creating '{target_output_folder}' folder")
            os.mkdir(target_output_folder)
        self.target_output_folder = target_output_folder

    def transform(self):
        import numpy as np
        import pickle

        bar = ProgBar(len(self.file_list),"Creating windows...")

        for data_path, target_path in zip(self.file_list, self.target_list):
            data = pickle.load(open(data_path, "rb"))
            target = pickle.load(open(target_path, "rb"))

            data_file_name = data_path.split('/')[-1].split('.p')[0]
            target_file_name = target_path.split('/')[-1].split('.p')[0]

            L, W = data.shape

            n_measures = int(np.max(data[:, 1]))
            n_ids = L//n_measures

            expanded_dimensions = data.reshape(n_ids, n_measures, W)
            
            del data

            for i, inicio in enumerate(range(0,n_ids - self.size, self.step)):
                data_window = expanded_dimensions[inicio:inicio+self.size, :, :]

                target_window = target[inicio:inicio+self.size]

                data_window = data_window.reshape(self.size*n_measures, W)

                pickle.dump(data_window,
                        open(self.data_output_folder + data_file_name + '__window_{}__.pkl'.format(i), "wb"), 
                            )

                pickle.dump(target_window,
                        open(self.target_output_folder + target_file_name + '__window_{}__.pkl'.format(i), "wb"), 
                            )
            
            if inicio + self.size != n_ids:
                i += 1
                data_window = expanded_dimensions[-self.size:, :, :]

                target_window = target[-self.size:]

                data_window = data_window.reshape(self.size*n_measures, W)

                pickle.dump(data_window,
                        open(self.data_output_folder + data_file_name + '__window_{}__.pkl'.format(i), "wb"), 
                            )

                pickle.dump(target_window,
                        open(self.target_output_folder + target_file_name + '__window_{}__.pkl'.format(i), "wb"), 
                            )
            
            bar.update()




from copyreg import pickle


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

    def __init__(self,input_folder,data_output_folder='iteration_data/', target_output_folder='iteration_target/',n_iterations = 33,maintained_percent=0.3):
        
        import glob

        self.file_list = glob.glob(input_folder+'*')
        self.file_list.sort()

        self.input_folder = input_folder
        self.target_output_folder = target_output_folder
        self.data_output_folder = data_output_folder
        self.maintained_percent = maintained_percent
        self.n_iterations = n_iterations

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

                aux,reduced_data = train_test_split(expanded_dimensions,test_size=self.maintained_percent,
                                                    shuffle=False)
                
                n_ids = len(np.unique(reduced_data[:,0]))
                
                reduced_data = reduced_data.reshape(len(reduced_data)*n_measures, 8)

                del data
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

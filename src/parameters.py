# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

class Parameters(metaclass=ABCMeta):
    def __init__(self):
        try:
            self.init_parameters()
        except:
            print('Skip add args')

    @abstractmethod
    def init_parameters(self):
        pass

    @abstractmethod
    def set_next_params(self):
        pass

    def get_title_from_params(self):

        members = [attr for attr in vars(self).items() ]

        param_str = []
        param_dict = {}
        used_words = []
        for member in members:
            name = member[0]
            value = member[1]

            param_name = ''
            for c in name:
                param_name += c
                if param_name not in used_words:
                    break

            used_words.append(param_name)
            param_dict.update({param_name : value})

            param_str.append( param_name + str(value) )

        return "-".join(param_str), param_dict

class LifeSavingResourcesParams(Parameters):
    def __init__(self):
        self._idx = 0
        self._num_of_grid_search = 0
        self.init_parameters()

    def init_parameters(self):
        self.range_patients = [4,7,14]
        #self.range_patients = [4,]
        self.range_fire_departments = [14,]
        self.range_rendezvous_points = [20,40,80,160,320,640,1280,2560]
        #self.range_rendezvous_points = [20,40]
        self.range_basehospitals = [14,]

    def set_next_params(self):
        if self._idx >= self._num_of_grid_search and self._idx != 0:
            return False

        if self._idx == 0:
            self._grid_search_params_list = []
            for patients in self.range_patients:
                for fire_departments in self.range_fire_departments:
                    for rendezvous_points in self.range_rendezvous_points:
                        for basehospitals in self.range_basehospitals:
                            self._grid_search_params_list.append([patients,fire_departments,rendezvous_points,basehospitals])
                            self._num_of_grid_search += 1

        params = self._grid_search_params_list[self._idx]
        self.patients = params[0]
        self.fire_departments = params[1]
        self.rendezvous_points = params[2]
        self.basehospitals = params[3]

        self._idx += 1
        return True
        

    def get_title_from_params(self):

        param_str = ['patients' + f'{self.patients:04}', \
                     'fire_departments' + str('{:.3f}'.format(self.fire_departments)), \
                     'rendezvous_points' + str('{:.3f}'.format(self.rendezvous_points)), \
                     'basehospitals' + str('{:.3f}'.format(self.basehospitals))]

        param_dict = {'patients' : self.patients, \
                      'fire_departments' : self.fire_departments, \
                      'rendezvous_points' : self.rendezvous_points, \
                      'basehospitals' : self.basehospitals}

        return "-".join(param_str), param_dict

class HyperParams(Parameters):
    def __init__(self):
        self.init_parameters()

    def init_parameters(self):
        self._idx = 0
        self._num_of_grid_search = 0
        self.lam1 = [40,]
        self.lam2 = [40,]
        self.lam3 = [-0.220,-0.222,-0.224,-0.226,-0.228,-0.229,-0.230]
        #self.lam3 = [0.24,0.25]

    def set_next_params(self):
        if self._idx >= self._num_of_grid_search and self._idx != 0:
            return False

        if self._idx == 0:
            self._grid_search_params_list = []
            for lam1 in self.lam1:
                for lam2 in self.lam2:
                    for lam3 in self.lam3:
                        self._grid_search_params_list.append([lam1,lam2,lam3])
                        self._num_of_grid_search += 1

        params = self._grid_search_params_list[self._idx]
        self.lam1 = params[0]
        self.lam2 = params[1]
        self.lam3 = params[2]

        self._idx += 1
        return True
        

    def get_title_from_params(self):

        param_str = ['lam1' + str('{:.3f}'.format(self.lam1)), \
                     'lam2' + str('{:.3f}'.format(self.lam2)), \
                     'lam3' + str('{:.3f}'.format(self.lam3))]

        param_dict = {'lam1' : self.lam1, \
                      'lam2' : self.lam2, \
                      'lam3' : self.lam3}

        return "-".join(param_str), param_dict


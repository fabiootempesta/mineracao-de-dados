#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import logging
from abc import abstractmethod


LOG = 'classifier.log'


class Classifier():
    def __init__(self, csv_file, debugging=False):
        '''
        Classifier template class
        '''
        #Flag de depuração
        self._debugging = debugging
        # Configura log
        self._log = logging.getLogger(LOG)
        # Lê arquivo CSV para _data
        self._data = pd.read_csv(csv_file, skipinitialspace=True)
        # Pega último atributo para classe
        self.class_att = str(self._data.columns[-1])
        # Demais atributos são atributos de dados
        self.att_list = list(self._data.columns[:-1])

    @abstractmethod
    def fit(self):
        '''
        Treinamento do classificador
        '''
        pass

    @abstractmethod
    def classify_record(self, record):
        '''
        Classifica um registro
        '''
        pass

    def _debug(selfself, mssg, *args, **kwargs):
        '''
        Exibe mensagens de debug
        '''
        self._log.debug(msg, *args, **kwargs)

    def _config_log(self):
        '''
        Configura o log
        '''
        # Cria log
        logger = logging.getLogger(LOG)
        # Formato de mensagens
        str_format = '%(levelname)s - %(message)s'
        log_format = logging.Formatter(str_format)
        # Arquivo
        file_handler = logging.FileHandler(LOG)
        file_handler.setFormatter(log_format)
        # Console
        console_hanler = logging.StreamHandler()
        console_hanler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logger.addHandler(console_hanler)
        # Nível de log
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def disable_debug(self):
        '''
        Desabilita o debug
        '''
        self._log.setLevel(logging.INFO)

    def precision(self, test_data):
        '''
        Calcula a precisão considerando os dados de test_data
        '''
        # Contagem de erros
        errors = 0
        # Para cada registro de dado
        for _, rec in test_data.iterrows():
            # Classifica o registro
            class_pre = self.classify_record(rec)
            # Compara com a classe correta do registro
            if class_pre != rec[self.class_att]:
                # Se for diferente conta o erro
                errors += 1
        # Retorna a porcentagem de acertos
        return (len(test_data) - errors) / len(test_data)

    def training_precision(self):
        '''
        Calcula a precisão usando todo o conjunto de treinamento
        '''
        return self.precision(self._data)

    def precision_k_fold(self, k):
        '''
        Calcula a precisão usando validação cruzada
        '''
        # Faz cópia dos dados originais
        bkp_data = self._data.copy(deep=True)
        # Tamanho de cada partição (fold)
        fold_len = len(bkp_data) // k
        precision = 0
        for cont in range(k):
            fold_start = cont * fold_len
            fold_end = (cont + 1) * fold_len
            test_data = bkp_data[fold_start:fold_end]
            train_data = bkp_data.drop([fold_start, fold_end - 1])
            self._data = train_data
            self.train()
            precision += self.precision(test_data)
        return precision / k

#!/usr/bin/python3
# -*- coding: utf-8 -*-

from classifier import Classifier


class DecisionTree(Classifier):
    
    def __init__(self, csv_file):
        Classifier.__init__(self, csv_file)
        self._tree = None

    def _split(self, data, att):
        '''
        Split list of records by values of att
        '''
        # Dicionário para guardar as divisões
        div_dict = {}
        # Contagem dos valores de att
        count_data = self.count_values(data, att)
        # Para cada valor de att
        for value in count_data.index:
            # Copia os dados originais
            new_data = data.copy(deep=True)
            # Seleciona apenas os dados para o valor atual de att
            new_data = new_data[new_data[att]==value]
            # Remove att dos dados
            new_data.drop(att, axis=1, inplace=True)
            # Incluir a partição no dicionário
            div_dict[value] = new_data
        # retorna o dicionário com as partições
        return div_dict
    
    def _entropy(self, data):
        '''
        Calc entropy of data
        '''
        from math import log
        # Log na base 2
        log2 = lambda x:log(x)/log(2)
        # Conta os valores da classe em data
        count_data = self.count_values(data, self.class_att)
        # Calcula entropia
        ent = 0.0
        for value in count_data:
            percent = value / len(data)
            ent = ent - percent * log2(percent)
        return ent    
    
    def _impurity(self, div_dict):
        '''
        Calc de impurity of div_dict
        '''
        # Calcula o número de registros em todas as partições de div_dict
        rec_num = 0
        for data in div_dict.values():
            rec_num += len(data)
        # Calcula impureza
        imp = 0
        for data in div_dict.values():
            imp += len(data) / rec_num * self._entropy(data)
        return imp
    
    
    def _best_att_info_gain(self, data):
        '''
        Calc de attribute with best information gain
        '''
        # Condição de parada (apenas um atributo e a classe)
        if len(data.columns) == 2:
            # Retorna o único atributo (exceto a classe)
            return data.columns[0]
        # Calcula entropia para os dados em data
        ent = self._entropy(data)
        self._log.debug('Current entropy: %f', ent)
        # Procura pelo atributo com melhor ganho de informação
        best_att = None
        best_gain = 0
        # Percorre todos os atributos nos dados (exceto a classe)
        for att in data.columns[:-1]:
            # Particiona os dados usando att
            div_dict = self._split(data, att)
            # Calcula impureza do particionamento atual
            imp = self._impurity(div_dict)
            # Calcula ganho de informação
            gain = ent - imp
            self._log.debug('Gain for attribute %s: %f', att, gain)
            # Testa se é o melhor ganho até o momento
            if gain > best_gain:
                # Atualiza melhor atributo e melhor ganho
                best_att = att
                best_gain = gain
        return best_att
    
    
    def _build_tree(self, data):
        '''
        Build tree recursively
        '''
        # Contagem de valores da calsse
        count_data = self.count_values(data, self.class_att)
        # Condições de parada:
        #   Único valor para a classe
        #   OU
        #   Não há mais atributos além da classe
        if len(count_data) == 1 or len(data.columns) == 1:
            # Retorna o valor de classe mais frequent
            frequent_class = count_data.index[0]
            return frequent_class
        else:
            # Seleciona atributo com melhor ganho de informação
            best_att = self._best_att_info_gain(data)
            # Particiona os dados usaddo o atributo selecionado
            div_dict = self._split(data, best_att)
            # Nós filhos a serem criado
            node_dict = {}
            # Cria um filho para cada valor do atributo
            for value in div_dict:
                # Dados com o valor atual
                new_data = div_dict[value]
                # Chamada recursiva
                child = self._build_tree(new_data)
                # Rótulo para o filho atual
                node_dict[value] = child
            # Return best_att com sua estrutura de filhos
            return {best_att: node_dict}

    def train(self):
        # Constrói a árvore
        self._tree = self._build_tree(self._data)

    def print_tree(self):
        '''
        Print the decision tree structure
        '''
        # Imprime atributo da raiz
        att = list(self._tree.keys())[0]
        print(att)
        # Imprime filhos
        self._print_node(att, self._tree[att], 1)

    def _print_node(self, father, node, level):
        # Para cada valor
        for value in node:
            # Obtém filho para o valor correspondente
            child = node[value]
            # Imprime o valor atual
            print('    '*level, '--- (', father, '=', value, ')', end='---> ')
            # Verifica se há filho
            if isinstance(child, dict):
                # Imprime o attrobuto do filho
                att = list(child.keys())[0]
                print(att)
                # Chamada recursiva para o filho
                self._print_node(value, child[att], level+1)
            else:
                # Imprime a folha
                print(self.class_att, '=', child)

    def classify_record(self, record):
        '''
        Classify a record
        '''
        return self._classify(self._tree, record)
    
    def _classify(self, node, record):
        '''
        Classify a record
        '''
        # Testa se o nó é não folha    
        if isinstance(node, dict):
            # Obtém attributo do nó
            keys = list(node.keys())
            att = keys[0]
            # Obtém valor do atributo no registro
            rec_value = record[att]
            # Obtém o filho correspondente
            new_node = node[att]
            new_node = new_node[rec_value]
            return self._classify(new_node, record)
        return node


def main():
    import argparse
    # Parâmetro -i para receber o arquivo CSV
    parser = argparse.ArgumentParser('Arvore')
    parser.add_argument('-i', '--input-file')
    args = parser.parse_args()
    if args.input_file:
        # Constrói a árvore usando o arquivo
        dec_tree = DecisionTree(args.input_file)
        dec_tree.log(True)
        dec_tree.train()
        dec_tree.log(False)
        # Imprime a estrutura da árvore
        print('\n\nDecision tree:')
        dec_tree.print_tree()
        # Imprime a precisão sobre o conjunto de treinamento
        print('Training precision:', dec_tree.training_precision())
        # Imprime a precisão para validação cruzada 10 folds
        print('Cross validation (10-fold) precision:', dec_tree.precision_k_fold(10))


if __name__ == '__main__':
    main()

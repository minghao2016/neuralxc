import numpy as np
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

class Formatter(TransformerMixin, BaseEstimator):

    def __init__(self, basis = None):
        """ Formatter allows to convert between the np.ndarray and the dictionary
        format of basis set representations
        """
        self._basis = basis
        self._rule = None

    def fit(self, C):
        """ Providing C in dictionary format, build set of rules for transformation
        """

        self._rule = {}
        for idx, key, data in expand(C):
            self._rule[key] = [s for s in data[0]]

    def transform(self, C):
        """ Transforms from a dictionary format ({n,l,m} : value)
         that is used internally by neuralxc to an ordered np.ndarray format
        """
        self.fit(C)
        transformed = [{}]*len(C)

        for idx, key, data in expand(C):
            data = data[0]
            if not key in transformed[idx]: transformed[idx][key] = []
            for d in data:
                transformed[idx][key].append(np.array([d[s] for s in d]))

            transformed[idx][key] = np.array(transformed[idx][key])
        if not isinstance(C, list):
            return transformed[0]
        else:
            return transformed

    def inverse_transform(self, C):
        """ Transforms from an ordered np.ndarray format to a dictionary
        format ({n,l,m} : value) that is used internally by neuralxc
        """
        transformed = [{}]*len(C)

        for idx, key, data in expand(C):
            data = data[0]
            if not key in transformed[idx]: transformed[idx][key] = []


            if not isinstance(self._rule, dict):
                basis = self._basis[key]
                rule = ['{},{},{}'.format(n,l,m) for n in range(0,basis['n'])  for l in range(0,basis['l']) for m in range(-l,l+1)]
            else:
                rule = self._rule[key]
            for d in data:
                transformed[idx][key].append(dict(zip(rule, d.tolist())))

        if not isinstance(C, list):
            return transformed[0]
        else:
            return transformed

class SpeciesGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, attrs, sys_species):
        """SpeciesGrouper allows to transform an array with columns
        [system_idx, feature1, feature2, ..., featureN, target]
        to be transformed into an representation that is grouped by atomic
        like so: [{'species1': np.ndarray(values), 'species2': np.ndarray(values)},
                  ...] where the outer list runs over independent systems
        (e.g. different molecules). The first representation has to be used if
        cross-validation should be performed while training. The second representation
        corresponds to the one used internally by neuralxc

        Parameters:
        -----------
        attrs: dict,
            basis set

        sys_species: list of strings
            per system, order of species inside system
        """
        self._attrs = attrs
        if not isinstance(sys_species, list):
            raise ValueError('sys_species must be a list but is {}'.format(sys_species))
        self._sys_species = sys_species

    def get_params(self, *args, **kwargs):
        return {'attrs' : self._attrs, 'sys_species': self._sys_species}

    def fit(self, X, y=None,**fit_params):
        return self

    def transform(self, X, y=None):
        """ Transform from ungrouped to grouped represenation
        """
        y = X[:,-1].real
        X = X[:,:-1]

        # First column should give system index
        if not isinstance(y,np.ndarray):
            y = self._y

        system = X[:,0]
        n_sys = np.max(system) + 1

        X = X[:,1:]

        features = []
        targets = []

        if not n_sys == len(self._sys_species):
            raise ValueError('Number of systems in X and len(sys_species) incompatible')

        for this_sys, _ in enumerate(self._sys_species):
            this_species = self._sys_species[this_sys]
            X_sys = X[system == this_sys]
            y_sys = y[system == this_sys]

            feat_dict = {}

            idx = 0
            for spec in this_species:
                if spec not in feat_dict:
                    feat_dict[spec] = []

                vec_len = self._attrs[spec]['n'] * sum([2*l+1 for l in range(self._attrs[spec]['l'])])

                feat_dict[spec].append(X_sys[:,idx:idx+vec_len])
                idx += vec_len

            for spec in feat_dict:
                feat_dict[spec] = np.array(feat_dict[spec]).swapaxes(0,1)

            features.append(feat_dict)
            targets.append(y_sys)

        return features, targets

    def get_gradient(self, X):
        # Required by NXCPipeline
        if isinstance(X, list):
            targets = [np.zeros(len(list(x.values())[0])) for x in X]
        else:
            targets = [np.zeros(len(list(X.values())[0]))]
        return self.inverse_transform(X,targets)

    def inverse_transform(self, features, targets):
        """ Transform from grouped to ungrouped representation
        """
        total_length = np.sum([len(tar) for tar in targets])
        max_vec_len = np.max([np.sum([feat[spec].shape[1]*feat[spec].shape[2] for spec in feat])\
                       for feat in features])


        X = np.zeros([total_length, max_vec_len + 1], dtype = complex)
        y = np.zeros(total_length)


        if not len(features) == len(targets) == len(self._sys_species):
            raise ValueError('number of systems inconsistent')

        sys_loc = 0
        for sysidx, (feat, tar) in enumerate(zip(features, targets)):
            this_species = self._sys_species[sysidx]
            this_len = len(tar)
            X[sys_loc:sys_loc+this_len, 0] = sysidx
            unique_species = np.unique([char for char in this_species])
            spec_loc = {spec: 0 for spec in unique_species}
            idx = 1

            for spec in this_species:
                insert = feat[spec][:,spec_loc[spec],:]
#                 print(insert.shape)
#                 print(X[sys_loc:sys_loc+this_len,idx:idx+insert.shape[-1]].shape)
                X[sys_loc:sys_loc+this_len,idx:idx+insert.shape[-1]] = insert
                spec_loc[spec] += 1
                idx += insert.shape[-1]


            y[sys_loc:sys_loc + this_len] = tar
            sys_loc += this_len

        return np.concatenate([X, y.reshape(-1,1)], axis = -1 )

def expand(*args):
    """ Takes the common format in which datasets such as D and C are provided
     (usually [{'species': np.ndarray}]) and loops over it
     """
    args = list(args)
    for i, arg in enumerate(args):
        if not isinstance(arg, list):
            args[i] = [arg]

    for idx, datasets in enumerate(zip(*args)):
        for key in datasets[0]:
            yield (idx, key , [data[key] for data in datasets])

def atomic_shape(X):
    return X.reshape(-1,X.shape[-1])

def system_shape(X, n):
    return X.reshape(-1,n,X.shape[-1])
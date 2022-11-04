

class Experiment:
    
    def __init__(
        self,
        target_columns,
        subset_columns,
        subset_columns_names,
        use_PRE_onlys,
        train_on_fulls,
        inclusion_criterias,
        cols_to_exclude):
        """
        Args:
            target_columns (list): List of target columns to use.
            subset_columns (list): List of subset columns to use.
            subset_columns_names (list): List of subset column names to use.
            use_PRE_onlys (list): List of bools to use PRE only or not.
            train_on_fulls (list): List of bools to train on full or not.
            inclusion_criterias (list): List of inclusion criteria to use.
            cols_to_exclude (list): List of columns to exclude.
            Note: len(subset_columns) == len(subset_columns_names)
        """
        assert len(subset_columns) == len(subset_columns_names), \
            "subset_columns and subset_columns_names must be the same."
        ret_target_columns, experiment_names, ret_subset_columns, ret_use_PRE_onlys, ret_train_on_fulls, ret_inclusion_criterias = \
            self._generate_experiments(
                target_columns,
                subset_columns,
                subset_columns_names,
                use_PRE_onlys,
                train_on_fulls,
                inclusion_criterias,
                cols_to_exclude)
        self.target_columns = ret_target_columns
        self.experiment_names = experiment_names
        self.subset_columns = ret_subset_columns
        self.use_PRE_onlys = ret_use_PRE_onlys
        self.train_on_fulls = ret_train_on_fulls
        self.inclusion_criterias = ret_inclusion_criterias
            
        
    def _generate_experiments(
        self, 
        target_columns, 
        subset_columns, 
        subset_columns_names, 
        use_PRE_onlys,
        train_on_fulls,
        inclusion_criterias,
        cols_to_exclude):
        """
        Generates experiments to run.
        Each experiment is a unique combination of the following Args:

        Args:
            target_columns (list): List of target columns to use.
            subset_columns (list): List of subset columns to use.
            subset_columns_names (list): List of subset column names to use.
            use_PRE_onlys (list): List of bools to use PRE columns only or not.
            train_on_fulls (list): List of bools to train on full or not.
            inclusion_criterias (list): List of inclusion criteria to use.
            cols_to_exclude (list): List of columns to exclude.
        Returns:
            target_columns (list): List of target columns.
            experiment_names (list): List of experiment names.
            subset_columns (list): List of subset columns.
            use_PRE_onlys (list): List of bools to use PRE only or not.
            train_on_fulls (list): List of bools to train on full or not.
            inclusion_criterias (list): List of inclusion criteria to use.
        """
        experiment_names = []
        ret_target_columns = []
        ret_subset_columns = []
        ret_use_PRE_onlys = [] 
        ret_train_on_fulls = []
        ret_inclusion_criterias = []
        for use_PRE_only, train_on_full, inclusion_criteria in zip(use_PRE_onlys, train_on_fulls, inclusion_criterias):
            for target_col in target_columns:
                for subset_col, subset_col_name in zip(subset_columns, subset_columns_names):
                    experiment_name = f"{target_col} {subset_col_name}_omit_{len(cols_to_exclude)} {inclusion_criteria} {'train_on_full' if train_on_full else ''}"
                    experiment_names.append(experiment_name)
                    ret_target_columns.append(target_col)
                    ret_subset_columns.append(subset_col)
                    ret_use_PRE_onlys.append(use_PRE_only)
                    ret_train_on_fulls.append(train_on_full)
                    ret_inclusion_criterias.append(inclusion_criteria)
        assert len(experiment_names) == len(ret_target_columns) == len(ret_subset_columns)
        print(f"Generated {len(experiment_names)} experiments using {len(target_columns)} target columns and {len(subset_columns)} subset columns.")
        return ret_target_columns, experiment_names, ret_subset_columns, ret_use_PRE_onlys, ret_train_on_fulls, ret_inclusion_criterias

    def __len__(self):
        return len(self.experiment_names)
    
    def __getitem__(self, idx):
        return self.target_columns[idx], self.experiment_names[idx], self.subset_columns[idx], self.use_PRE_onlys[idx]
    
    def __iter__(self):
        # Returns an iterator of dictionaries, each dictionary item is an attribute and its value
        for idx in range(len(self)):
            yield {
                "target_column": self.target_columns[idx],
                "experiment_name": self.experiment_names[idx],
                "subset_cols": self.subset_columns[idx],
                "use_PRE_only": self.use_PRE_onlys[idx],
                "train_on_full": self.train_on_fulls[idx],
                "inclusion_criteria": self.inclusion_criterias[idx]
            }
        # return iter(zip(self.target_columns, self.experiment_names, self.subset_columns, self.use_PRE_onlys))
    
    # def __repr__(self):
    #     return f"{self.__class__.__name__}({self.target_columns},{self.experiment_names},  {self.subset_columns}, {self.use_PRE_onlys})"
    
    # def __str__(self):
    #     return f"{self.__class__.__name__}({self.target_columns},{self.experiment_names},  {self.subset_columns}, {self.use_PRE_onlys})"
    
    def __eq__(self, other):
        return self.target_columns == other.target_columns and self.subset_columns == other.subset_columns and \
            self.experiment_names == other.experiment_names and self.use_PRE_onlys == other.use_PRE_onlys and \
            self.train_on_fulls == other.train_on_fulls and self.inclusion_criterias == other.inclusion_criterias
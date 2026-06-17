export const h = {
  button: {
    dataset: {
      upload: 'Import Dataset',
      download: 'Export Dataset',
      delete: 'Delete Dataset'
    },
    pic: {
      sign_data: 'Label',
      choose_machine: 'Select Device',
      sign_error: 'As Abnormal',
      sign_correct: 'As Normal',
      as_sign: 'As Target'
    }
  },
  topic: {
    dataset: 'Dataset',
    upload: 'Raw Dataset',
    result: 'Result Dataset',
    plot: 'Scatter Plot Label',
    line: 'Line Chart Label'
  },
  words: {
    dataset: {
      name: 'Dataset Name',
      type: 'Type',
      id: 'Entity ID',
      metric: 'Metric',
      result: 'Result Files',
      method: 'Algorithm'
    },
    pic: {
      data: 'Data Selection',
      method: 'Algorithm Selection',
      category: 'Category Selection',
      machine: 'Machine Selection',
      choose_team: 'Current Team Selection',
      please_choose: 'Please Select',
      cluster: 'Select Cluster',
      please_input: 'Please Enter Selection',
      choose_item: 'Selected Points',
      time_x: 'Timeline',
      value_y: ' Value',
      scatter_x: 'Dimension 1',
      scatter_y: 'Dimension 2',
      time: 'time:',
      cluster_id: 'cluster:'
    }
  },
  message: {
    backstage: 'Backend Error',
    not_choose_data_method: 'Dataset or algorithm not selected, please check',
    not_choose_range: 'No annotation range selected',
    not_choose_category_machine: 'No category or machine selected',
    not_match: 'Modified information does not match, please confirm',
    choose_machine: 'Please select the annotation machine first',
    choose_error: 'Selection is abnormal, please check before selecting',
    too_many_plot: 'Too many points selected, please process them first',
    too_many_cluster: 'Exceeds selectable clusters, currently can handle a maximum of 8 clusters or less',
    change_success: 'Backend modification successful',
    name_error: 'File naming is incorrect, please name it ending with _metric',
    metric_name_error: 'Metric naming format is incorrect, please check',
    exist_file: 'The file already exists, please change the file name',
    save_success: 'Save successful',
    no_rawdata: 'No corresponding raw file, please input and retry',
    upload_raw_not_match: 'Uploaded result dataset does not match the original dataset',
    name_error_res: 'Naming is incorrect, please check the suffix',
    no_target: 'No target file, please check',
    delete_success: 'Delete successful',
    no_data: 'No such file, please upload manually'
  }
}

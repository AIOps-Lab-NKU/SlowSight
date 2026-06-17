<template>
  <div class="dashboard-container">
    <!-- Top section for dataset and method selection -->
    <div style="display: flex;justify-content: space-evenly;margin-top: 2%">
      <div style="display: flex;align-items: center">
        <div>{{ $t('h.words.pic.data') }}：</div>
        <el-select
          v-model="dataset"
          :placeholder="$t('h.words.pic.please_choose')"
          filterable
        >
          <el-option
            v-for="item in datasetSelect"
            :key="item"
            :label="item"
            :value="item"
          />
        </el-select>
      </div>
      <div style="display: flex;align-items: center">
        <div>{{ $t('h.words.pic.method') }}：</div>
        <el-select
          v-model="method"
          :placeholder="$t('h.words.pic.please_choose')"
          filterable
        >
          <el-option
            v-for="item in methodSelect"
            :key="item"
            :label="item"
            :value="item"
          />
        </el-select>
      </div>
      <el-button
        type="primary"
        class="icon-button"
        @click="predictTarget"
      >{{ $t('h.button.pic.sign_data') }}</el-button>
      <!--      <el-button type="primary" class="icon-button" @click="saveTarget">保存标注</el-button>-->
    </div>
    <!-- Scatter plot annotation section -->
    <div style="height: 90%;overflow-y: auto;">
      <div
        v-show="showItemSelect"
        style="height: 100%;margin-top: 2%"
      >
        <div
          v-for="(item,index) in dataList"
          :key="index"
          style="height: 75%;display: flex; justify-content: space-evenly"
        >
          <!-- Scatter plot display section -->
          <div
            :id="'scatter' + index"
            style="height: 100%;width: 75%"
          />
          <!-- Annotation operation buttons -->
          <div style="width: 20%;text-align: center;">
            <el-button
              type="primary"
              class="icon-button"
              style="width: 70%;margin-bottom: 10px;"
              @click="setStatus(true, index)"
            >{{ $t('h.button.pic.sign_error') }}</el-button>
            <div
              style="display: flex;width: 100%;justify-content: space-between;align-items: center;"
            >
              <span>{{ $t('h.words.pic.cluster') }}</span>
              <el-input
                v-model="item['selectCluster']"
                :placeholder="$t('h.words.pic.please_input')"
                style="width: 40%"
              />
              <el-button
                type="primary"
                class="icon-button"
                @click="setStatus(false, index)"
              >{{ $t('h.button.pic.as_sign') }}</el-button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
// import { mapGetters } from 'vuex'
import * as echarts from 'echarts'
// import * as XLSX from 'xlsx'
// import ecStat from 'echarts-stat'

export default {
  data() {
    return {
      dataList: [], // Store image information from the backend
      COLOR_ALL: [
        '#d07f7a', '#92b7c8', '#a89181', '#6fa49d',
        '#cdc3e3', '#93445e', '#6381ab', '#ceb3aa',
        '#aed09c', '#8974ba', '#efbbd2', '#306692',
        '#c19382', '#7f9f6f', '#5a5472', '#cb7e99',
        '#8db1e1', '#a5796b', '#527865', '#b882af'
      ], // Color selection for scatter plot annotations
      symbolAll: [
        'path://M975.069283 746.176494a162.756163 162.756163 0 0 1-230.134685 230.134686L511.97742 743.399162 279.065402 976.31118A162.756163 162.756163 0 0 1 48.930717 746.176494L281.842734 513.241896 48.930717 280.329879a162.688423 162.688423 0 0 1 0-230.112106 162.711003 162.711003 0 0 1 230.134685 0l232.912018 232.889438L744.912018 50.217773a162.711003 162.711003 0 1 1 230.134685 230.112106L742.134686 513.241896l232.934597 232.934598z',
        'circle', 'rect', 'triangle', 'diamond', 'pin', 'arrow',
        'path//M781.186088 616.031873q17.338645 80.573705 30.59761 145.848606 6.119522 27.537849 11.219124 55.075697t9.689243 49.976096 7.649402 38.247012 4.079681 19.888446q3.059761 20.398406-9.179283 27.027888t-27.537849 6.629482q-5.099602 0-14.788845-3.569721t-14.788845-5.609562l-266.199203-155.027888q-72.414343 42.836653-131.569721 76.494024-25.498008 14.278884-50.486056 28.557769t-45.386454 26.517928-35.187251 20.398406-19.888446 10.199203q-10.199203 5.099602-20.908367 3.569721t-19.378486-7.649402-12.749004-14.788845-2.039841-17.848606q1.01992-4.079681 5.099602-19.888446t9.179283-37.737052 11.729084-48.446215 13.768924-54.055777q15.298805-63.23506 34.677291-142.788845-60.175299-52.015936-108.111554-92.812749-20.398406-17.338645-40.286853-34.167331t-35.697211-30.59761-26.007968-22.438247-11.219124-9.689243q-12.239044-11.219124-20.908367-24.988048t-6.629482-28.047809 11.219124-22.438247 20.398406-10.199203l315.155378-28.557769 117.290837-273.338645q6.119522-16.318725 17.338645-28.047809t30.59761-11.729084q10.199203 0 17.848606 4.589641t12.749004 10.709163 8.669323 12.239044 5.609562 10.199203l114.231076 273.338645 315.155378 29.577689q20.398406 5.099602 28.557769 12.239044t8.159363 22.438247q0 14.278884-8.669323 24.988048t-21.928287 26.007968z',
        'path://M51.911,16.242C51.152,7.888,45.239,1.827,37.839,1.827c-4.93,0-9.444,2.653-11.984,6.905 c-2.517-4.307-6.846-6.906-11.697-6.906c-7.399,0-13.313,6.061-14.071,14.415c-0.06,0.369-0.306,2.311,0.442,5.478 c1.078,4.568,3.568,8.723,7.199,12.013l18.115,16.439l18.426-16.438c3.631-3.291,6.1'
      ], // Symbol selection for scatter plot annotations
      selectedList: [], // List to store selected node information
      chooseIndex: null, // Store the index of the chart being operated on
      chooseItem: null, // Store the instance of the chart being operated on
      dataset: null, // Selected dataset
      datasetSelect: [], // List of available dataset options
      datasetInfo: [], // Store internal information of datasets
      methodSelect: [], // List of available method options
      method: null, // Selected method
      showItemSelect: false // Flag to show item selection
    }
  },
  mounted() {
    this.getBase()
  },
  methods: {
    setPic(data, index) {
      console.log(data)
      const chartDom = document.getElementById('scatter' + index)
      const myChart = echarts.init(chartDom)
      const dataWithCluster = data.data
      const nameList = data.nameList
      const COLOR_DIMENSION = 4
      const SYMBOL_DIMENSION = 2
      var pieces = []
      for (var i = 0; i < nameList.length; i++) {
        pieces.push({
          value: i,
          label: nameList[i],
          color: this.COLOR_ALL[i]
        })
      }
      const that = this
      var option = {
        dataset: [
          {
            source: dataWithCluster
          }
        ],
        brush: {
          id: '1', // Component ID
          z: 1, // z-index of the brush area
          xAxisIndex: 0, // Specify the x-axis index for the brush tool
          yAxisIndex: 1,
          toolbox: ['rect'],
          brushLink: [1, 2],
          brushType: 'rect', // Default brush type
          brushMode: 'single', // Default brush mode
          transformable: true, // Whether the brush area can be moved
          throttleType: 'fixRate', // Update mode for data animation when the brush area changes
          throttleDelay: 100, // Delay for data animation updates when the brush area changes
          removeOnClick: true, // Whether single click clears all brush areas
          autoBrush: true,
          // Brush area style
          brushStyle: {
            color: 'red', // Color of the brush area
            borderColor: 'red', // Border color of the brush area
            borderWidth: 5, // Border width of the brush area
            borderType: 'solid', // Border type of the brush area
            borderDashOffset: 5, // Offset for dashed border
            borderCap: 'butt', // Shape of the line ends
            borderJoin: 'bevel', // Shape of the line joins
            borderMiterLimit: 10, // Miter limit for bevel joins
            shadowColor: 'red', // Shadow color of the brush area
            opacity: 0.1 // Opacity of the brush area
          },
          outOfBrush: {
            opacity: 0.5 // Opacity of elements outside the brush area
          }
        },
        tooltip: {
          position: 'top',
          formatter: function(params) {
            // Custom content for the tooltip
            return 'X:' + params.data[0] + ' Y:' + params.data[1] + ' ID:' + params.data[2] + ' time:' + params.data[3]
          }
        },
        visualMap: {
          type: 'piecewise',
          top: 'middle',
          min: 0,
          max: 1,
          right: 0,
          splitNumber: 1,
          dimension: COLOR_DIMENSION,
          pieces: pieces
        },
        grid: {
          left: '5%',
          right: '23%'
        },
        xAxis: {
          name: this.$t('h.words.pic.scatter_x'),
          nameTextStyle: {
            fontSize: 18 // Set X-axis name font size
          }
        },
        yAxis: {
          name: this.$t('h.words.pic.scatter_y'),
          nameTextStyle: {
            fontSize: 18 // Set Y-axis name font size
          }
        },
        series: {
          type: 'scatter',
          encode: {
            x: 0, // Specify x-axis using the 0th dimension of data
            y: 1 // Specify y-axis using the 1st dimension of data
          },
          symbolSize: 15,
          itemStyle: {
            borderColor: '#555'
          },
          datasetIndex: 0,
          symbol: function(params) {
            return that.symbolAll[params[SYMBOL_DIMENSION] + 1]
          }
        },
        dataZoom: [
          {
            type: 'inside', // Use mouse wheel for zooming
            xAxisIndex: [0], // Affects only x-axis
            start: 0,
            end: 100
          }
        ]
      }
      myChart.setOption(option)
      myChart.on('brushEnd', function(params) {
        console.log('brushEnd', params)
        that.selectedList = []
        if (!params.areas[0]) return
        that.chooseIndex = index
        that.chooseItem = myChart
        const range = params.areas[0]?.coordRange
        // Rectangle selection
        if (params.areas[0]?.brushType === 'rect') {
          // Rectangle coordinates
          const rangeX = range[0]
          const rangeY = range[1]
          for (let i = 0; i < dataWithCluster.length; i++) {
            // Point coordinates
            const x = dataWithCluster[i][0]
            const y = dataWithCluster[i][1]
            // Check if the point is within the selected rectangle area
            if ((rangeX[0] <= x && x <= rangeX[1]) && (rangeY[0] <= y && y <= rangeY[1])) {
              that.selectedList.push(i)
            }
          }
        }
        console.log('selectedList', that.selectedList)
      })

      // Add event listener to resize the chart when the window size changes
      window.addEventListener('resize', () => {
        myChart.resize()
      })
    },
    // Method to set the status of selected points
    setStatus(ifError, index) {
      if (index !== this.chooseIndex) {
        this.$message({
          type: 'warning',
          message: this.$t('h.message.not_match')
        })
        return // Show a warning message if the selected index does not match the current index
      }
      if (this.selectedList.length < 1) {
        this.$message({
          type: 'warning',
          message: this.$t('h.message.not_choose_range')
        })
        return // Show a warning message if no points are selected
      }
      console.log('here')
      const selectData = this.dataList[this.chooseIndex].data // Get the data of the selected chart
      const selectNameList = this.dataList[this.chooseIndex].nameList // Get the name list of the selected chart
      const changeList = [] // Initialize the list to store changes
      var changeCluster = -1 // Initialize the cluster to change
      if (ifError) {
        console.log(this.selectedList)
        this.selectedList.forEach((item) => {
          console.log(selectData)
          selectData[item][2] = -1 // Mark the selected points as error by setting the cluster to -1
          changeList.push([selectData[item][3], selectNameList[selectData[item][4]]]) // Add the point to the change list
        })
      } else {
        if (!this.dataList[this.chooseIndex].selectCluster) {
          this.$message({
            type: 'warning',
            message: this.$t('h.message.not_match')
          })
        }
        if (parseInt(this.dataList[this.chooseIndex].selectCluster) >= this.symbolAll.length) {
          this.$message({
            type: 'warning',
            message: this.$t('h.message.too_many_cluster')
          })
          return // Show a warning message if the selected cluster is out of range
        }
        changeCluster = this.dataList[this.chooseIndex].selectCluster // Set the cluster to change
        this.selectedList.forEach((item) => {
          selectData[item][2] = parseInt(changeCluster) // Update the cluster of the selected points
          changeList.push([selectData[item][3], selectNameList[selectData[item][4]]]) // Add the point to the change list
        })
      }
      this.selectedList = [] // Clear the selected list
      this.changePredict(changeList, changeCluster) // Send the changes to the server
      this.chooseItem.setOption({ dataset: [{ source: selectData }] }, { notMerge: false }) // Update the chart with the new data
      this.chooseItem.dispatchAction({
        type: 'brush',
        areas: []
      }) // Clear the brush selection
    },
    // Method to fetch base data for datasets and methods
    getBase() {
      const url = this.$baseUrl + '/getBase'
      this.$http.post(url, { type: 'scatter' })
        .then((res) => {
          console.log(res)
          if (res.status === 200) {
            this.datasetInfo = res.data.dataset // Store dataset information
            this.datasetSelect = Object.keys(res.data.dataset) // Create a list of dataset options
            console.log(this.datasetSelect)
            this.methodSelect = res.data.method // Store method options
          } else {
            this.$message({
              type: 'warning',
              message: this.$t('h.message.backstage')
            })
          }
        })
        .catch((error) => {
          console.log(error)
          this.$message({
            type: 'warning',
            message: error
          })
        })
    },

    // Method to predict target data using the selected dataset and method
    predictTarget() {
      const url = this.$baseUrl + '/dbscanPredict'
      if (this.dataset == null || this.method == null) {
        this.$message({
          type: 'warning',
          message: this.$t('h.message.not_choose_data_method')
        })
        return // Show a warning message if dataset or method is not selected
      }
      const dict = {
        dataset: this.dataset,
        method: this.method
      }
      this.$http.post(url, dict)
        .then((res) => {
          console.log(res)
          if (res.status === 200) {
            if (res.data.status === 'warning') {
              this.$message({
                type: res.data.status,
                message: this.$t(res.data.info)
              })
              return // Show a warning message if the response status is warning
            }
            this.showItemSelect = true // Enable item selection
            this.dataList = res.data.option // Store the predicted data
            console.log('data', this.dataList)
            this.$nextTick(() => {
              this.dataList.forEach((item, index) => {
                this.setPic(item, index) // Render scatter plots for each data item
              })
            })
          } else {
            this.$message({
              type: 'warning',
              message: this.$t('h.message.backstage')
            })
          }
        })
        .catch((error) => {
          console.log(error)
          this.$message({
            type: 'warning',
            message: error
          })
        })
    },

    // Method to change the prediction based on selected points and cluster
    changePredict(changeList, targetCluster) {
      const url = this.$baseUrl + '/dbChangePredict'
      const dict = {
        fileName: this.dataset,
        changeList: changeList,
        targetCluster: targetCluster,
        method: this.method
      }
      this.$http.post(url, dict)
        .then((res) => {
          console.log('changePredict', res)
          if (res.status === 200) {
            this.$message({
              type: 'success',
              message: this.$t('h.message.change_success')
            })
          } else {
            this.$message({
              type: 'warning',
              message: this.$t('h.message.backstage')
            })
          }
        })
        .catch((error) => {
          console.log(error)
          this.$message({
            type: 'warning',
            message: error
          })
        })
    }

  }
}
</script>
<style scoped>
.dashboard-container {
  height: 90vh;
}
/deep/.el-input__inner {
  border: black solid 2px;
}
/deep/.el-select .el-input__inner:hover {
  border: #20a0ff solid 2px;
}
/deep/.el-select .el-input__inner::placeholder {
  color: #383737; /* Customize placeholder color */
}
/deep/ .el-select .el-input__suffix .el-select__caret {
  color: #383737; /* Customize arrow color */
}
/deep/ .icon-button {
  height: 50px;
}
</style>

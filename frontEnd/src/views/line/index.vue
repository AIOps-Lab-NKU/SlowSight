<template>
  <div class="dashboard-container">
    <!-- Top section for algorithm and dataset selection -->
    <div style="display: flex;justify-content: space-evenly;margin-top: 2%">
      <div style="display: flex;align-items: center">
        <div>{{ $t('h.words.pic.data') }}：</div>
        <el-select
          v-model="dataset"
          :placeholder="$t('h.words.pic.please_choose')"
          filterable
          @change="changeSelect1"
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
    <div style="height: 90%; overflow-y: auto;margin-top: 2%">
      <!-- Bottom section for entity category and machine selection -->
      <div
        v-show="showItemSelect"
        style="display: flex;justify-content: space-evenly; margin-bottom: 20px"
      >
        <div style="display: flex;align-items: center">
          <div>{{ $t('h.words.pic.category') }}：</div>
          <el-select
            v-model="category"
            :placeholder="$t('h.words.pic.please_choose')"
            filterable
            @change="changeSelect2"
          >
            <el-option
              v-for="item in categorySelect"
              :key="item"
              :label="item"
              :value="item"
            />
          </el-select>
        </div>
        <div style="display: flex;align-items: center">
          <div>{{ $t('h.words.pic.machine') }}：</div>
          <el-select
            v-model="machine"
            :placeholder="$t('h.words.pic.please_choose')"
            filterable
            multiple
            collapse-tags
          >
            <el-option
              v-for="item in machineSelect"
              :key="item"
              :label="item"
              :value="item"
            />
          </el-select>
        </div>
        <el-button
          type="primary"
          class="icon-button"
          @click="chooseMachine"
        >{{ $t('h.button.pic.choose_machine') }}</el-button>
      </div>
      <!-- Loop to generate line charts for all metrics -->
      <div
        v-for="(item, index) in dataList"
        :key="item['picName']"
        style="height: 80%;width: 100%;display: flex;justify-content: space-evenly"
      >
        <!-- Line chart settings -->
        <div
          :id="item['picName']"
          style="height: 100%;width: 70%"
        />
        <!-- Line chart annotation operation buttons -->
        <div style="width: 25%">
          <div style="display: flex;justify-content: space-evenly">
            <el-button
              type="primary"
              class="icon-button"
              @click="setStatus(1, index)"
            >{{ $t('h.button.pic.sign_error') }}</el-button>
            <el-button
              type="primary"
              class="icon-button"
              @click="setStatus(0, index)"
            >{{ $t('h.button.pic.sign_correct') }}</el-button>
          </div>
          <div
            style="display: flex;align-items: center;justify-content: center;margin-top: 20px"
          >
            <div>{{ $t('h.words.pic.choose_team') }}：</div>
            <el-select
              ref="test"
              v-model="item['chooseTeam']"
              :placeholder="$t('h.words.pic.please_choose')"
              filterable
            >
              <el-option
                v-for="(team, i) in item['eachTeam']"
                :key="i"
                :label="i + '.' + team['name']"
                :value="i + '.' + team['name']"
              />
            </el-select>
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
// import Papa from 'papaparse'

export default {
  data() {
    return {
      dataset: null, // Selected dataset
      datasetSelect: [], // List of dataset options
      datasetInfo: [], // Internal information of datasets
      methodSelect: [], // List of method options
      method: null, // Selected method
      showItemSelect: false, // Flag to show entity type and machine selection
      category: null, // Selected category
      categorySelect: [], // List of category options
      machine: [], // Selected machines
      machineSelect: [], // List of machine options
      changeList: [], // List to record modification intervals
      chooseItem: null, // ECharts instance of the selected metric
      chooseIndex: null, // Index of the selected metric
      chooseSeriesIndex: null, // Index of the selected machine line
      selectColor: [
        '#aed09c', '#8974ba', '#efbbd2', '#306692',
        '#f6948e', '#92b7c8', '#a89181', '#8cbdb6',
        '#b6a6dd', '#77374c', '#7396c6', '#ceb3aa',
        '#c19382', '#7f9f6f', '#8b84ae', '#cb7e99',
        '#8db1e1', '#a5796b', '#527865', '#b882af'
      ], // Colors for line charts
      dataList: [] // Dataset for ECharts charts
    }
  },
  mounted() {
    this.getBase()
  },
  methods: {
    // Update category options based on selected dataset
    changeSelect1() {
      this.category = ''
      this.machine = []
      this.categorySelect = Object.keys(this.datasetInfo[this.dataset])
      console.log('changeSelect1', this.categorySelect)
    },
    // Update machine options based on selected dataset and category
    changeSelect2() {
      this.machine = []
      this.machineSelect = this.datasetInfo[this.dataset][this.category]['ids'].sort()
    },
    // Set annotation status {ifError: whether the annotation is an error; index: index of the annotation chart}
    setStatus(ifError, index) {
      if (parseInt(this.chooseIndex, 10) !== index) {
        this.$message({
          type: 'warning',
          message: this.$t('h.message.not_match')
        })
        return
      }
      this.changePredict(ifError)
      this.changeList = []
    },
    // Set ECharts chart information, including initialization and click events on ECharts
    setPic(id, time, team, index) {
      const chartDom = document.getElementById(id)
      if (chartDom) {
        var myChart_temp = echarts.getInstanceByDom(chartDom)
        if (myChart_temp) {
          myChart_temp.dispose()
        }
      }
      const myChart = echarts.init(chartDom)

      var option = this.getOption(id, time, team)
      console.log(option)
      const that = this
      myChart.getZr().on('click', function(params) {
        console.log('param', params)
        const pointInPixel = [params.offsetX, params.offsetY]
        if (myChart.containPixel('grid', pointInPixel)) {
          if (that.dataList[index].chooseTeam.length < 1) {
            that.$message({
              type: 'warning',
              message: that.$t('h.message.choose_machine')
            })
            return
          }
        } else {
          return
        }
        if (that.chooseIndex !== index && that.chooseIndex !== null) {
          that.changeList = []
          option = that.getOption(that.dataList[that.chooseIndex].picName, that.dataList[that.chooseIndex].time, that.dataList[that.chooseIndex].eachTeam)
          myChart.setOption(option)
        }
        that.chooseItem = myChart
        that.chooseIndex = index
        that.chooseSeriesIndex = parseInt(that.dataList[index].chooseTeam.split('.', 2)[0])
        var xIndex = null
        if (myChart.containPixel('grid', pointInPixel)) {
          const pointInGrid = myChart.convertFromPixel({ seriesIndex: 0 }, pointInPixel)
          xIndex = pointInGrid[0]
          console.log('区域点击数据', xIndex, team, that.chooseSeriesIndex)
        } else {
          that.$message({
            type: 'warning',
            message: that.$t('h.message.choose_error')
          })
          return
        }
        if (that.changeList.includes(xIndex)) {
          that.changeList = that.changeList.filter(item => item !== xIndex)
          option.series[that.chooseSeriesIndex].markPoint.data = option.series[that.chooseSeriesIndex].markPoint.data.filter(item => item.name !== xIndex)
          myChart.setOption({ series: option.series }, { notMerge: false })
        } else if (that.changeList.length < 2) {
          option = that.changeList.length < 1 ? that.getOption(id, that.dataList[index].time,
            that.dataList[index].eachTeam) : option
          that.changeList.push(xIndex)
          option.series[that.chooseSeriesIndex].markPoint.data.push({
            name: xIndex,
            coord: [xIndex, team[that.chooseSeriesIndex].data[xIndex]],
            itemStyle: { color: 'blue' },
            label: { show: true, formatter: that.$t('h.words.pic.choose_item') + (that.changeList.length - 1) }
          })
          // Update chart
          myChart.setOption({ series: option.series }, { notMerge: false })
        } else {
          that.$message({
            type: 'warning',
            message: that.$t('h.message.too_many_plot')
          })
        }
      })
      myChart.setOption(option)

      window.addEventListener('resize', () => {
        myChart.resize()
      })
    },
    // Construct error intervals as red areas in the chart
    setPieces(data) {
      const result = []
      for (const i in data) {
        const dict = {
          gt: data[i][0] - 1,
          lte: data[i][1]
        }
        result.push(dict)
      }
      if (result.length < 1) {
        result.push({ gt: -2, lte: -1 })
      }
      return result
    },
    // Set basic information for ECharts chart
    getOption(topic, time, team) {
      const visual = []
      const series = []
      const name = []
      var maxZoom = (4000 / time.length).toFixed(2) * 100
      maxZoom = maxZoom > 100 ? 100 : maxZoom
      console.log(maxZoom)
      for (const i in team) {
        name.push(team[i].name)
        // Construct color mapping dictionary
        const visualDict = {
          show: false,
          dimension: 0,
          pieces: this.setPieces(team[i].error),
          inRange: {
            color: ['red']
          },
          outOfRange: {
            color: [this.selectColor[i]]
          },
          seriesIndex: i
        }
        visual.push(visualDict)
        // Construct information for each line
        const seriesDict = {
          name: team[i].name,
          type: 'line',
          smooth: true,
          markPoint: {
            data: []
          },
          data: team[i].data
        }
        series.push(seriesDict)
      }
      return {
        title: {
          text: topic
        },
        tooltip: {
          trigger: 'axis'
        },
        color: this.selectColor,
        legend: {
          orient: 'vertical',
          right: 0,
          top: 'middle',
          data: name
        },
        grid: {
          containLabel: true,
          right: '20%',
          left: '5%'
        },
        xAxis: {
          name: this.$t('h.words.pic.time_x'),
          nameTextStyle: {
            fontSize: 18 // Set X-axis name font size
          },
          type: 'category',
          boundaryGap: false,
          data: time
        },
        yAxis: {
          name: topic + this.$t('h.words.pic.value_y'),
          nameLocation: 'end',
          nameGap: 20,
          nameTextStyle: {
            fontSize: 18, // Set Y-axis name font size
            align: 'left'
          },
          type: 'value',
          axisLabel: {
            formatter: '{value}'
          },
          axisPointer: {
            snap: true
          }
        },
        visualMap: visual,
        series: series,
        dataZoom: [
          {
            type: 'slider', // Use slider for zooming
            show: true, // Show slider
            xAxisIndex: [0], // Affects only x-axis
            start: 0, // Initial zoom range
            end: 100, // Initial zoom range
            minSpan: 0, // Minimum zoom range (percentage value), range is 0 ~ 100.
            maxSpan: maxZoom
          },
          {
            type: 'inside', // Use mouse wheel for zooming
            xAxisIndex: [0], // Affects only x-axis
            start: 0,
            end: 100
          }
        ]
      }
    },
    // Fetch basic information of the page, including datasets and methods
    getBase() {
      const url = this.$baseUrl + '/getBase'
      this.$http.post(url, { type: 'line' })
        .then((res) => {
          console.log(res)
          if (res.status === 200) {
            this.datasetInfo = res.data.dataset
            this.datasetSelect = Object.keys(res.data.dataset)
            console.log(this.datasetSelect)
            this.methodSelect = res.data.method
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
    // Send annotation request to the backend based on selected dataset and method
    predictTarget() {
      const url = this.$baseUrl + '/multi/predictTarget'
      if (this.dataset == null || this.method == null) {
        this.$message({
          type: 'warning',
          message: this.$t('h.message.not_choose_data_method')
        })
        return
      }
      this.category = null
      this.machine = []
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
              return
            }
            this.showItemSelect = true
            this.dataList = res.data.option
            this.$nextTick(() => {
              this.dataList.forEach((item, index) => {
                this.setPic(item.picName, item.time, item.eachTeam, index)
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
    // Send request to the backend to annotate selected interval as error or correct
    changePredict(ifError) {
      const url = this.$baseUrl + '/multi/changePredict'
      if (this.chooseIndex == null || this.changeList.length < 2) {
        this.$message({
          type: 'warning',
          message: this.$t('h.message.not_choose_range')
        })
        return
      }
      const dict = {
        fileName: this.dataset,
        metricName: this.dataList[this.chooseIndex].picName,
        targetID: this.dataList[this.chooseIndex].chooseTeam.split('.', 2)[0],
        targetName: this.dataList[this.chooseIndex].eachTeam[this.chooseSeriesIndex].name,
        changeRange: this.changeList,
        ifError: ifError,
        method: this.method
      }
      this.$http.post(url, dict)
        .then((res) => {
          console.log('changePredict', res)
          if (res.status === 200) {
            this.dataList[this.chooseIndex]['eachTeam'][this.chooseSeriesIndex].error = res.data.errorList
            const option = this.getOption(this.dataList[this.chooseIndex].picName,
              this.dataList[this.chooseIndex].time, this.dataList[this.chooseIndex].eachTeam)
            this.chooseItem.setOption({ series: option.series, visualMap: option.visualMap }, { notMerge: false })
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
    // Send request to the backend to choose specific machines
    chooseMachine() {
      const url = this.$baseUrl + '/multi/chooseMachine'
      console.log(this.category, this.machine)
      if (this.category === '' || this.machine.length < 1) {
        this.$message({
          type: 'warning',
          message: this.$t('h.message.not_choose_category_machine')
        })
        return
      }
      const dict = {
        dataset: this.dataset,
        categoryName: this.category,
        itemList: this.machine.map(String),
        method: this.method
      }
      console.log('chooseMachine', dict)
      this.$http.post(url, dict)
        .then((res) => {
          console.log('chooseMachine', res)
          if (res.status === 200) {
            this.dataList = res.data.option
            this.$nextTick(() => {
              this.dataList.forEach((item, index) => {
                this.setPic(item.picName, item.time, item.eachTeam, index)
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
</style>

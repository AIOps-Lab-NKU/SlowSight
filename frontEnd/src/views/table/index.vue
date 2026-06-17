<template>
  <div class="app-container">
    <!-- Button section for file upload, download, and delete -->
    <div style="display: flex;justify-content: space-evenly;margin-bottom: 20px">
      <el-button
        type="primary"
        class="icon-button"
        @click="selectXlsxFile1"
      >
        <span class="button-text">{{ $t('h.button.dataset.upload') }}</span>
      </el-button>
      <input
        ref="xlsxInput1"
        type="file"
        accept=".csv"
        style="display: none;"
        @change="handleFileChange1"
      >
      <el-button
        type="primary"
        class="icon-button"
        @click="downloadFile"
      >
        <span class="button-text">{{ $t('h.button.dataset.download') }}</span>
      </el-button>
      <el-button
        type="danger"
        class="icon-button"
        @click="deleteFiles"
      >
        <span class="button-text">{{ $t('h.button.dataset.delete') }}</span>
      </el-button>
    </div>
    <!-- Table for displaying dataset information -->
    <el-table
      v-if="dataType===0"
      :data="list"
      :span-method="objectSpanMethod"
      element-loading-text="Loading"
      border
      fit
      stripe
      highlight-current-row
      :header-cell-style="{ 'font-size': '1.3em' }"
      :cell-style="{ 'font-size': '1.2em' }"
      @selection-change="handleSelectionChange"
    >
      <el-table-column
        align="center"
        type="selection"
        width="55"
      />
      <el-table-column
        align="center"
        label=""
        type="index"
        width="95"
      />
      <el-table-column
        :label="$t('h.words.dataset.name')"
        align="center"
      >
        <template slot-scope="scope">
          {{ scope.row.name }}_metric
        </template>
      </el-table-column>
      <el-table-column
        :label="$t('h.words.dataset.type')"
        width="110"
        align="center"
      >
        <template slot-scope="scope">
          {{ scope.row.category }}
        </template>
      </el-table-column>
      <el-table-column
        :label="$t('h.words.dataset.id')"
        align="center"
      >
        <template slot-scope="scope">
          {{ scope.row.ids }}
        </template>
      </el-table-column>
      <el-table-column
        class-name="status-col"
        :label="$t('h.words.dataset.metric')"
        align="center"
        class="table_item"
      >
        <template slot-scope="scope">
          {{ scope.row.metrics }}
        </template>
      </el-table-column>
    </el-table>
    <!-- Table for displaying result information -->
    <el-table
      v-else
      :data="list"
      element-loading-text="Loading"
      border
      fit
      stripe
      highlight-current-row
      :header-cell-style="{ 'font-size': '1.3em' }"
      :cell-style="{ 'font-size': '1.2em' }"
      @selection-change="handleSelectionChange"
    >
      <el-table-column
        align="center"
        type="selection"
        width="55"
      />
      <el-table-column
        align="center"
        label=""
        type="index"
        width="95"
      />
      <el-table-column
        :label="$t('h.words.dataset.name')"
        align="center"
      >
        <template slot-scope="scope">
          {{ scope.row.name }}
        </template>
      </el-table-column>
      <el-table-column
        :label="$t('h.words.dataset.method')"
        align="center"
      >
        <template slot-scope="scope">
          {{ scope.row.method_name }}
        </template>
      </el-table-column>
      <el-table-column
        :label="$t('h.words.dataset.result')"
        align="center"
      >
        <template slot-scope="scope">
          <div
            v-for="(item, index) in scope.row.result_name"
            :key="index"
          >
            <span>{{ item }}</span>
          </div>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>
<script>
import Papa from 'papaparse'

export default {
  filters: {
    statusFilter(status) {
      const statusMap = {
        published: 'success',
        draft: 'gray',
        deleted: 'danger'
      }
      return statusMap[status]
    }
  },
  data() {
    return {
      list: null, // List to store table data
      listLoading: true, // Loading state for the table
      dataType: null, // Type of data to display (0 or other)
      multipleSelection: [], // List of selected rows
      mergeList: [] // List to store row merge information
    }
  },
  mounted() {
    this.getBase() // Fetch base data when the component is mounted
  },
  created() {
    // this.fetchData() // Commented out fetchData method
    this.dataType = this.$route.meta.type // Set dataType based on route meta
    console.log(this.$route.meta)
  },
  methods: {
    // Handle selection change in the table
    handleSelectionChange(val) {
      console.log('选择框', val)
      this.multipleSelection = val
    },
    // Get dataset list from the dataset object
    getDatasetList(dataset) {
      return Object.keys(dataset).flatMap(host => {
        const data = dataset[host]
        return Object.keys(data).map(category => {
          return {
            name: host,
            category: category,
            ids: data[category].ids.join(','),
            metrics: data[category].metrics.join(',')
          }
        })
      })
    },
    // Get result list from the result object
    getResultList(result) {
      return Object.keys(result).flatMap(dataset => {
        const data = result[dataset]
        return Object.keys(data).map(method => {
          return {
            name: dataset,
            method_name: method,
            result_name: data[method]
          }
        })
      })
    },
    // Get span array for row merging
    getSpanArr(data) {
      let count = 0 // Start position for row merging
      this.mergeList = []
      data.forEach((item, index) => {
        if (index === 0) {
          this.mergeList.push(1)
        } else {
          if (item['name'] === data[index - 1]['name']) {
            this.mergeList[count] += 1
            this.mergeList.push(0)
          } else {
            count = index
            this.mergeList.push(1)
          }
        }
      })
      console.log(this.mergeList)
    },
    // Object span method for row merging in the table
    objectSpanMethod({ row, column, rowIndex, columnIndex }) {
      if (columnIndex === 1 || columnIndex === 0 || columnIndex === 2) {
        if (this.mergeList[rowIndex]) {
          return [this.mergeList[rowIndex], 1]
        } else {
          return [0, 0]
        }
      }
    },
    // Trigger file selection dialog
    selectXlsxFile1() {
      this.$refs.xlsxInput1.click()
    },
    // Handle file change event
    handleFileChange1(event) {
      const files = event.target.files
      if (files.length) {
        const file = files[0]
        const reader = new FileReader()

        reader.onload = (e) => {
          const data = e.target.result
          const parsedData = Papa.parse(data, {
            header: true,
            skipEmptyLines: true
          })
          const jsonData = parsedData.data
          const dict = {
            name: file.name,
            data: jsonData,
            dataType: this.dataType
          }
          const url = this.$baseUrl + '/upload'
          this.$http.post(url, dict)
            .then((res) => {
              console.log(res)
              const result = res.data
              this.$message({
                type: result.status,
                message: this.$t(result.info)
              })
              this.list = this.dataType === 0 ? this.getDatasetList(result.dataset) : this.getResultList(result.dataset)
              this.getSpanArr(this.list)
            })
            .catch((error) => {
              console.log(error)
            })
        }
        reader.readAsBinaryString(file)
      }
    },
    // Download selected files
    downloadFile() {
      this.multipleSelection.forEach((item) => {
        if (this.dataType === 0) {
          this.downloadItem(item.name, 'metric', null)
        } else {
          item.result_name.forEach((item2) => {
            this.downloadItem(item.name, item2, item.method_name)
          })
        }
      })
    },
    // Download a single item
    downloadItem(name, itemName, methodName) {
      const url = this.$baseUrl + '/download'
      this.$http.post(url, { responseType: 'blob', name: name, dataType: this.dataType, itemName: itemName, method: methodName })
        .then(response => {
          const url = window.URL.createObjectURL(new Blob([response.data]))
          const link = document.createElement('a')
          link.href = url
          link.setAttribute('download', `${name + '_' + itemName}.csv`)
          document.body.appendChild(link)
          link.click()
          document.body.removeChild(link)
          window.URL.revokeObjectURL(url)
        })
        .catch(error => {
          console.error(error)
        })
    },
    // Delete selected files
    deleteFiles() {
      const url = this.$baseUrl + '/delete'
      this.$http.post(url, {
        deleteList: Object.values(this.multipleSelection).map(item => item.name),
        methodList: Object.values(this.multipleSelection).map(item => item.method_name),
        dataType: this.dataType
      })
        .then((res) => {
          console.log(res)
          if (res.data.status === 'warning') {
            this.$message({
              type: res.data.status,
              message: this.$t(res.data.info)
            })
            return
          }
          this.$message({
            type: res.data.status,
            message: this.$t(res.data.info)
          })
          this.list = this.dataType === 0 ? this.getDatasetList(res.data.dataset) : this.getResultList(res.data.dataset)
          this.getSpanArr(this.list)
          console.log(this.list)
        })
        .catch((error) => {
          this.$message({
            type: 'warning',
            message: error
          })
          console.log(error)
        })
    },
    // Fetch base data based on dataType
    getBase() {
      if (this.dataType === 0) {
        const url = this.$baseUrl + '/getBase'
        this.$http.post(url, { type: 'line' })
          .then((res) => {
            this.list = this.getDatasetList(res.data.dataset)
            this.getSpanArr(this.list)
            console.log('after change', this.list)
          })
          .catch((error) => {
            this.$message({
              type: 'warning',
              message: error
            })
            console.log(error)
          })
      } else {
        const url = this.$baseUrl + '/getResult'
        this.$http.get(url)
          .then((res) => {
            console.log(res)
            this.list = this.getResultList(res.data.dataset)
          })
          .catch((error) => {
            this.$message({
              type: 'warning',
              message: error
            })
            console.log(error)
          })
      }
    }
  }
}
</script>

<style>
.table_item {
  display: flex;
  white-space: normal;
  word-wrap: break-word;
  justify-content: center;
  overflow: auto;
}
</style>

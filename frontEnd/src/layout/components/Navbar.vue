<template>
  <div class="navbar">
    <hamburger
      :is-active="sidebar.opened"
      class="hamburger-container"
      @toggleClick="toggleSideBar"
    />

    <breadcrumb class="breadcrumb-container" />
    <el-button
      type="primary"
      class="right-menu"
      @click="changeLangEvent()"
    >
      {{ language }}
    </el-button>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
import Breadcrumb from '@/components/Breadcrumb'
import Hamburger from '@/components/Hamburger'

export default {
  components: {
    Breadcrumb,
    Hamburger
  },
  data() {
    return {
      language: 'EN'
    }
  },
  computed: {
    ...mapGetters([
      'sidebar',
      'avatar'
    ])
  },
  methods: {
    toggleSideBar() {
      this.$store.dispatch('app/toggleSideBar')
    },
    changeLangEvent() {
      console.log(this.$i18n)
      if (this.language === 'EN') {
        localStorage.setItem('locale', 'en')
        this.$i18n.locale = localStorage.getItem('locale')
        this.$message({
          message: 'Already switch to English',
          type: 'success'
        })
        localStorage.setItem('lang', 'EN')
        this.language = 'CN'
      } else if (this.language === 'CN') {
        localStorage.setItem('locale', 'ch')
        this.$i18n.locale = localStorage.getItem('locale')
        this.$message({
          message: '已切换为中文',
          type: 'success'
        })
        localStorage.setItem('lang', 'CN')
        this.language = 'EN'
      }
    }
  }
}
</script>

<style lang="scss" scoped>
.navbar {
  height: 50px;
  overflow: hidden;
  position: relative;
  background: #fff;
  box-shadow: 0 1px 4px rgba(0, 21, 41, 0.08);

  .hamburger-container {
    line-height: 46px;
    height: 100%;
    float: left;
    cursor: pointer;
    transition: background 0.3s;
    -webkit-tap-highlight-color: transparent;

    &:hover {
      background: rgba(0, 0, 0, 0.025);
    }
  }

  .breadcrumb-container {
    float: left;
  }

  .right-menu {
    float: right;
    height: 100%;
    //line-height: 50px;
    font-size: 1.3em;
    align-items: center;
  }
}
</style>

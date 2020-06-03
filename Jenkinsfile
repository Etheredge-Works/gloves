pipeline {
  agent {
    dockerfile {
      filename 'Dockerfile'
    }

  }
  stages {
    stage('Unit Testing') {
      parallel {
        stage('Unit Testing') {
          steps {
            echo 'Testing Utils'
            sh 'python -m pytest -n 8 test_utils.py'
          }
        }

        stage('') {
          steps {
            sh 'python -m pytest test_data.py'
          }
        }

      }
    }

  }
}
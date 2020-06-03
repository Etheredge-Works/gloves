pipeline {
  agent {
    dockerfile {
      filename 'Dockerfile'
    }

  }
  stages {
    stage('Unit Testing') {
      steps {
        echo 'Testing Utils'
        sh 'python -m pytest -n 32 test_utils.py'
      }
    }

  }
}
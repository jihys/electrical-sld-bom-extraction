// ──────────────────────────────────────────────────────────────────
// Cosmos DB – Serverless account with SLD BOM database & containers
// ──────────────────────────────────────────────────────────────────

@description('Cosmos DB account name')
param accountName string

@description('Azure region')
param location string

@description('Resource tags')
param tags object

var databaseName = 'sld-bom'

// ── Account (Serverless) ──────────────────────────────────────────
resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' = {
  name: accountName
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    disableLocalAuth: false
    capabilities: [
      { name: 'EnableServerless' }
    ]
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
  }
}

// ── Database ──────────────────────────────────────────────────────
resource database 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2024-05-15' = {
  parent: cosmosAccount
  name: databaseName
  properties: {
    resource: {
      id: databaseName
    }
  }
}

// ── Container: pipeline-runs ──────────────────────────────────────
resource pipelineRuns 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  parent: database
  name: 'pipeline-runs'
  properties: {
    resource: {
      id: 'pipeline-runs'
      partitionKey: {
        paths: ['/run_id']
        kind: 'Hash'
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        automatic: true
        includedPaths: [
          { path: '/*' }
        ]
        excludedPaths: [
          { path: '/"_etag"/?' }
        ]
      }
    }
  }
}

// ── Container: step-states ────────────────────────────────────────
resource stepStates 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  parent: database
  name: 'step-states'
  properties: {
    resource: {
      id: 'step-states'
      partitionKey: {
        paths: ['/run_id']
        kind: 'Hash'
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        automatic: true
        includedPaths: [
          { path: '/*' }
        ]
        excludedPaths: [
          { path: '/"_etag"/?' }
        ]
      }
    }
  }
}

// ── Container: file-cache ─────────────────────────────────────────
resource fileCache 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  parent: database
  name: 'file-cache'
  properties: {
    resource: {
      id: 'file-cache'
      partitionKey: {
        paths: ['/pdf_hash']
        kind: 'Hash'
      }
      defaultTtl: 2592000 // 30 days TTL
      indexingPolicy: {
        indexingMode: 'consistent'
        automatic: true
        includedPaths: [
          { path: '/*' }
        ]
        excludedPaths: [
          { path: '/"_etag"/?' }
        ]
      }
    }
  }
}

// ── Outputs ───────────────────────────────────────────────────────
output endpoint string = cosmosAccount.properties.documentEndpoint
output accountName string = cosmosAccount.name
output databaseName string = databaseName

// ──────────────────────────────────────────────────────────────────
// Storage Account – Blob containers for pipeline artifacts & cache
// ──────────────────────────────────────────────────────────────────

@description('Storage account name (3-24 chars, lowercase alphanumeric)')
@minLength(3)
@maxLength(24)
param storageAccountName string

@description('Azure region')
param location string

@description('Resource tags')
param tags object

// ── Storage Account ───────────────────────────────────────────────
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// ── Blob Service ──────────────────────────────────────────────────
resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: 7
    }
  }
}

// ── Container: pipeline-artifacts ─────────────────────────────────
resource pipelineArtifacts 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'pipeline-artifacts'
  properties: {
    publicAccess: 'None'
  }
}

// ── Container: file-cache ─────────────────────────────────────────
resource fileCache 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'file-cache'
  properties: {
    publicAccess: 'None'
  }
}

// ── Lifecycle Management (auto-delete old cache) ──────────────────
resource lifecyclePolicy 'Microsoft.Storage/storageAccounts/managementPolicies@2023-05-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    policy: {
      rules: [
        {
          name: 'delete-old-cache'
          enabled: true
          type: 'Lifecycle'
          definition: {
            filters: {
              blobTypes: ['blockBlob']
              prefixMatch: ['file-cache/']
            }
            actions: {
              baseBlob: {
                delete: {
                  daysAfterModificationGreaterThan: 30
                }
              }
            }
          }
        }
      ]
    }
  }
}

// ── Outputs ───────────────────────────────────────────────────────
output storageAccountName string = storageAccount.name
output blobEndpoint string = storageAccount.properties.primaryEndpoints.blob
output pipelineArtifactsContainer string = pipelineArtifacts.name
output fileCacheContainer string = fileCache.name
output storageAccountId string = storageAccount.id

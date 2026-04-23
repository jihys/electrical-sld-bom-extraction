// ──────────────────────────────────────────────────────────────────
// App Service – Linux container for Streamlit HITL UI
// ──────────────────────────────────────────────────────────────────

@description('App Service name')
param appName string

@description('Azure region')
param location string

@description('Resource tags')
param tags object

@description('App Service Plan SKU')
@allowed(['B1', 'B2', 'B3', 'S1', 'S2', 'P1v3', 'P2v3'])
param skuName string = 'B2'

@description('Cosmos DB endpoint for env config')
param cosmosEndpoint string = ''

@description('Cosmos DB database name')
param cosmosDatabase string = 'sld-bom'

@description('Blob Storage endpoint for env config')
param storageBlobEndpoint string = ''

@description('Storage account name')
param storageAccountName string = ''

@description('Service Bus endpoint')
param serviceBusEndpoint string = ''

@description('Azure OpenAI endpoint')
param azureOpenAiEndpoint string = ''

@description('Azure OpenAI deployment name')
param azureOpenAiDeployment string = 'gpt-5.4'

@description('Azure OpenAI API version')
param azureOpenAiApiVersion string = '2025-03-01-preview'

@description('Azure Document Intelligence endpoint')
param azureDiEndpoint string = ''

// ── App Service Plan (Linux) ──────────────────────────────────────
resource appServicePlan 'Microsoft.Web/serverfarms@2023-12-01' = {
  name: '${appName}-plan'
  location: location
  tags: tags
  kind: 'linux'
  sku: {
    name: skuName
  }
  properties: {
    reserved: true // Linux
  }
}

// ── Web App ───────────────────────────────────────────────────────
resource webApp 'Microsoft.Web/sites@2023-12-01' = {
  name: appName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: appServicePlan.id
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.12'
      appCommandLine: '/home/site/wwwroot/startup.sh'
      alwaysOn: true
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
      appSettings: [
        // ── Storage Mode ──
        { name: 'ENABLE_PERSISTENT_STATE', value: 'true' }
        // ── Azure OpenAI ──
        { name: 'AZURE_OPENAI_ENDPOINT', value: azureOpenAiEndpoint }
        { name: 'AZURE_OPENAI_DEPLOYMENT', value: azureOpenAiDeployment }
        { name: 'AZURE_OPENAI_API_VERSION', value: azureOpenAiApiVersion }
        // ── Azure Document Intelligence ──
        { name: 'AZURE_DI_ENDPOINT', value: azureDiEndpoint }
        { name: 'AZURE_DI_MODEL_ID', value: 'prebuilt-layout' }
        // ── Cosmos DB ──
        { name: 'AZURE_COSMOS_ENDPOINT', value: cosmosEndpoint }
        { name: 'AZURE_COSMOS_DATABASE', value: cosmosDatabase }
        // ── Blob Storage ──
        { name: 'AZURE_STORAGE_ACCOUNT', value: storageAccountName }
        { name: 'AZURE_STORAGE_BLOB_ENDPOINT', value: storageBlobEndpoint }
        { name: 'AZURE_STORAGE_ARTIFACTS_CONTAINER', value: 'pipeline-artifacts' }
        { name: 'AZURE_STORAGE_CACHE_CONTAINER', value: 'file-cache' }
        // ── Service Bus ──
        { name: 'AZURE_SERVICEBUS_NAMESPACE', value: serviceBusEndpoint }
        // ── Pipeline ──
        { name: 'CHECKPOINT_DIR', value: '/tmp/checkpoints' }
        { name: 'OUTPUT_DIR', value: '/tmp/outputs' }
        { name: 'HITL_CONFIDENCE_THRESHOLD', value: '0.7' }
        { name: 'GRID_SIZE', value: '120' }
        { name: 'VERIFY_MAX_TRIES', value: '10' }
        // ── Python ──
        { name: 'SCM_DO_BUILD_DURING_DEPLOYMENT', value: 'false' }
        { name: 'WEBSITES_PORT', value: '8000' }
        { name: 'WEBSITES_CONTAINER_START_TIME_LIMIT', value: '600' }
      ]
    }
  }
}

// ── Startup command script ────────────────────────────────────────
// Oryx build is DISABLED – startup.sh installs deps into a persistent
// venv at /home/site/venv on first boot (~10 min cold start on B2).
// Subsequent restarts skip pip install (hash-based marker file).

// ── Outputs ───────────────────────────────────────────────────────
output appServiceName string = webApp.name
output appServiceUrl string = 'https://${webApp.properties.defaultHostName}'
output appServicePlanName string = appServicePlan.name
output principalId string = webApp.identity.principalId

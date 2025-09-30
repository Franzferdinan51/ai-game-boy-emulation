/**
 * Configuration Service - Handles dynamic port allocation and service discovery
 */
interface ServiceConfig {
  name: string;
  port: number;
  url: string;
  status: 'unknown' | 'available' | 'unavailable';
}

interface SystemConfig {
  services: Record<string, ServiceConfig>;
  validation: {
    valid: boolean;
    missing_required: string[];
    missing_optional: string[];
    warnings: string[];
    api_keys_configured: number;
  };
  timestamp: number;
}

interface ConfigResponse {
  configuration: SystemConfig;
  status: number;
}

class ConfigService {
  private configCache: SystemConfig | null = null;
  private lastFetch: number = 0;
  private readonly CACHE_DURATION = 30000; // 30 seconds

  /**
   * Get service configuration from backend
   */
  async getConfiguration(baseUrl: string): Promise<SystemConfig> {
    const now = Date.now();

    // Return cached config if still valid
    if (this.configCache && (now - this.lastFetch) < this.CACHE_DURATION) {
      return this.configCache;
    }

    try {
      const response = await fetch(`${baseUrl}/api/config`);
      if (!response.ok) {
        throw new Error(`Failed to get configuration: ${response.status}`);
      }

      const data: SystemConfig = await response.json();
      this.configCache = data;
      this.lastFetch = now;

      return data;
    } catch (error) {
      console.error('Error fetching configuration:', error);
      throw error;
    }
  }

  /**
   * Validate system configuration
   */
  async validateConfiguration(baseUrl: string): Promise<ConfigResponse> {
    try {
      const response = await fetch(`${baseUrl}/api/config/validate`);
      const data = await response.json();

      return {
        configuration: data.configuration,
        status: response.status
      };
    } catch (error) {
      console.error('Error validating configuration:', error);
      throw error;
    }
  }

  /**
   * Get backend URL with dynamic port
   */
  getBackendUrl(config: SystemConfig): string {
    const backend = config.services.backend;
    if (!backend) {
      return 'http://localhost:5000'; // fallback
    }
    return backend.url;
  }

  /**
   * Get frontend URL with dynamic port
   */
  getFrontendUrl(config: SystemConfig): string {
    const frontend = config.services.frontend;
    if (!frontend) {
      return 'http://localhost:5173'; // fallback
    }
    return frontend.url;
  }

  /**
   * Get GLM UI URL with dynamic port
   */
  getGlmUiUrl(config: SystemConfig): string {
    const glmUi = config.services.glm_ui;
    if (!glmUi) {
      return 'http://localhost:3000'; // fallback
    }
    return glmUi.url;
  }

  /**
   * Get monitor URL with dynamic port
   */
  getMonitorUrl(config: SystemConfig): string {
    const monitor = config.services.monitor;
    if (!monitor) {
      return 'http://localhost:8080'; // fallback
    }
    return monitor.url;
  }

  /**
   * Check if backend is available
   */
  async checkBackendAvailability(baseUrl: string): Promise<boolean> {
    try {
      const response = await fetch(`${baseUrl}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      return response.ok;
    } catch (error) {
      console.error('Backend health check failed:', error);
      return false;
    }
  }

  /**
   * Get service status
   */
  async getServiceStatus(baseUrl: string): Promise<any> {
    try {
      const response = await fetch(`${baseUrl}/api/status`);
      if (!response.ok) {
        throw new Error(`Failed to get status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error getting service status:', error);
      throw error;
    }
  }

  /**
   * Test service connectivity
   */
  async testServiceConnectivity(serviceUrl: string): Promise<boolean> {
    try {
      const response = await fetch(`${serviceUrl}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(3000) // 3 second timeout
      });
      return response.ok;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get optimal backend URL from multiple sources
   */
  async getOptimalBackendUrl(): Promise<string> {
    // Try common URLs in order of preference
    const candidates = [
      'http://localhost:5000',
      'http://127.0.0.1:5000',
      'http://localhost:5001',
      'http://127.0.0.1:5001'
    ];

    for (const url of candidates) {
      if (await this.testServiceConnectivity(url)) {
        return url;
      }
    }

    // If no URL works, try to get dynamic configuration
    try {
      const config = await this.getConfiguration('http://localhost:5000');
      return this.getBackendUrl(config);
    } catch (error) {
      console.warn('Could not get dynamic configuration, using fallback');
      return candidates[0]; // fallback to first candidate
    }
  }

  /**
   * Get configuration summary for display
   */
  getConfigurationSummary(config: SystemConfig): {
    totalServices: number;
    availableServices: number;
    hasValidConfig: boolean;
    apiKeyCount: number;
    warnings: string[];
    serviceUrls: Record<string, string>;
  } {
    const services = Object.values(config.services);
    const availableServices = services.filter(s => s.status === 'available').length;

    return {
      totalServices: services.length,
      availableServices,
      hasValidConfig: config.validation.valid,
      apiKeyCount: config.validation.api_keys_configured,
      warnings: config.validation.warnings,
      serviceUrls: {
        backend: this.getBackendUrl(config),
        frontend: this.getFrontendUrl(config),
        glm_ui: this.getGlmUiUrl(config),
        monitor: this.getMonitorUrl(config)
      }
    };
  }

  /**
   * Clear configuration cache
   */
  clearCache(): void {
    this.configCache = null;
    this.lastFetch = 0;
  }

  /**
   * Get cached configuration or fetch if needed
   */
  async getCachedConfiguration(baseUrl: string): Promise<SystemConfig> {
    return this.getConfiguration(baseUrl);
  }
}

export const configService = new ConfigService();
export type { ServiceConfig, SystemConfig, ConfigResponse };
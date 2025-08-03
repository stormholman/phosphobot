import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Play, Square, RefreshCw, Eye, EyeOff, Key } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface SimulationStatus {
  status: "ok" | "error";
  message: string;
}

interface KinematicsStatus {
  status: "ok" | "error";
  message: string;
}

const SimulationPage: React.FC = () => {
  const [simulationStatus, setSimulationStatus] = useState<SimulationStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [kinematicsStatus, setKinematicsStatus] = useState<KinematicsStatus | null>(null);
  const [isKinematicsLoading, setIsKinematicsLoading] = useState(false);
  const [kinematicsError, setKinematicsError] = useState<string | null>(null);
  const [aiTask, setAiTask] = useState<string>("red cup"); // AI task input state
  
  // API Key management state
  const [apiKey, setApiKey] = useState<string>("");
  const [showApiKey, setShowApiKey] = useState<boolean>(false);
  const [apiKeySet, setApiKeySet] = useState<boolean>(false);
  const [isSettingApiKey, setIsSettingApiKey] = useState<boolean>(false);

  const checkSimulationStatus = async () => {
    console.log("🔍 Checking simulation status...");
    setError(null);
    try {
      const response = await fetch("http://localhost:80/simulation/status");
      console.log("📡 Status response:", response.status, response.statusText);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("📊 Status data:", data);
      setSimulationStatus(data);
    } catch (error) {
      console.error("❌ Failed to check simulation status:", error);
      setError(`Failed to check status: ${error}`);
    }
  };

  const launchSimulation = async () => {
    console.log("🚀 Launch simulation button clicked");
    setIsLoading(true);
    setError(null);
    try {
      console.log("📡 Making POST request to http://localhost:80/simulation/launch");
      const response = await fetch("http://localhost:80/simulation/launch", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json',
        },
      });
      console.log("📡 Response received:", response.status, response.statusText);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("📊 Response data:", data);
      setSimulationStatus(data);
      
      // Check status after a short delay to get updated info
      setTimeout(checkSimulationStatus, 1000);
    } catch (error) {
      console.error("❌ Failed to launch simulation:", error);
      setError(`Failed to launch simulation: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  const stopSimulation = async () => {
    console.log("🛑 Stop simulation button clicked");
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch("http://localhost:80/simulation/stop", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("📊 Stop response data:", data);
      setSimulationStatus(data);
      
      // Check status after a short delay to get updated info
      setTimeout(checkSimulationStatus, 1000);
    } catch (error) {
      console.error("❌ Failed to stop simulation:", error);
      setError(`Failed to stop simulation: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  /* ======================  API KEY MANAGEMENT  ====================== */
  const setAnthropicApiKey = async () => {
    if (!apiKey.trim()) {
      setKinematicsError("Please enter a valid API key");
      return;
    }

    setIsSettingApiKey(true);
    setKinematicsError(null);
    try {
      const response = await fetch("http://localhost:80/kinematics/set-api-key", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ api_key: apiKey.trim() }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("📊 API key response:", data);
      setApiKeySet(true);
      setKinematicsError(null);
      
      // Clear the input for security
      setApiKey("");
      setShowApiKey(false);
    } catch (err) {
      console.error("❌ Failed to set API key:", err);
      setKinematicsError(`Failed to set API key: ${err}`);
    } finally {
      setIsSettingApiKey(false);
    }
  };

  const checkApiKeyStatus = async () => {
    try {
      const response = await fetch("http://localhost:80/kinematics/api-key-status");
      if (response.ok) {
        const data = await response.json();
        setApiKeySet(data.api_key_set || false);
      }
    } catch (err) {
      console.error("Failed to check API key status:", err);
    }
  };

  /* ======================  KINEMATICS API  ====================== */
  const checkKinematicsStatus = async () => {
    console.log("🔍 Checking kinematics status...");
    setKinematicsError(null);
    try {
      const response = await fetch("http://localhost:80/kinematics/status");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setKinematicsStatus(data);
    } catch (err) {
      console.error("❌ Failed to check kinematics status:", err);
      setKinematicsError(`Failed to check status: ${err}` as string);
    }
  };

  const launchKinematics = async (mode: "manual" | "ai") => {
    console.log(`🚀 Launch kinematics (${mode}) button clicked`);
    
    // Check API key for AI mode
    if (mode === "ai" && !apiKeySet) {
      setKinematicsError("Please set your Anthropic API key first for AI mode");
      return;
    }
    
    // Validate AI task input for AI mode
    if (mode === "ai" && !aiTask.trim()) {
      setKinematicsError("Please enter a task description for AI mode");
      return;
    }
    
    setIsKinematicsLoading(true);
    setKinematicsError(null);
    try {
      const url = `http://localhost:80/kinematics/launch?mode=${mode}`;
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: mode === "ai" ? JSON.stringify({ task: aiTask.trim() }) : undefined,
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setKinematicsStatus(data);
      // refresh status after short delay
      setTimeout(checkKinematicsStatus, 1000);
    } catch (err) {
      console.error("❌ Failed to launch kinematics:", err);
      setKinematicsError(`Failed to launch kinematics: ${err}` as string);
    } finally {
      setIsKinematicsLoading(false);
    }
  };

  const stopKinematics = async () => {
    console.log("🛑 Stop kinematics button clicked");
    setIsKinematicsLoading(true);
    setKinematicsError(null);
    try {
      const response = await fetch("http://localhost:80/kinematics/stop", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setKinematicsStatus(data);
      setTimeout(checkKinematicsStatus, 1000);
    } catch (err) {
      console.error("❌ Failed to stop kinematics:", err);
      setKinematicsError(`Failed to stop kinematics: ${err}` as string);
    } finally {
      setIsKinematicsLoading(false);
    }
  };

  // Check status on component mount (both simulation and kinematics)
  useEffect(() => {
    console.log("🎬 SimulationPage component mounted – checking both statuses");
    checkSimulationStatus();
    checkKinematicsStatus();
    checkApiKeyStatus();
  }, []);

  const isRunning = simulationStatus?.message?.toLowerCase().includes("running") && 
                   !simulationStatus?.message?.toLowerCase().includes("no simulation");
  console.log("🔍 isRunning:", isRunning, "message:", simulationStatus?.message);

  const isKinematicsRunning =
    kinematicsStatus?.message?.toLowerCase().includes("running") &&
    !kinematicsStatus?.message?.toLowerCase().includes("no kinematics");

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <header className="bg-white shadow-md p-4">
        <h1 className="text-2xl font-bold">Simulations &amp; AI Kinematics</h1>
      </header>
      <main className="flex-grow p-4">
        <Tabs defaultValue="simulation">
          <TabsList className="grid w-full grid-cols-2 mb-4">
            <TabsTrigger value="simulation">MuJoCo</TabsTrigger>
            <TabsTrigger value="kinematics">AI Kinematics</TabsTrigger>
          </TabsList>

          {/* -------------- MuJoCo Tab -------------- */}
          <TabsContent value="simulation">
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <RefreshCw className="h-5 w-5" />
                    Simulation Control
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {error && (
                    <div className="p-3 bg-red-100 border border-red-400 text-red-700 rounded">
                      Error: {error}
                    </div>
                  )}
                  
                  <div className="flex items-center gap-4">
                    <Button
                      onClick={launchSimulation}
                      disabled={isLoading || isRunning}
                      className="flex items-center gap-2"
                    >
                      <Play className="h-4 w-4" />
                      Launch Simulation
                    </Button>
                    <Button
                      onClick={stopSimulation}
                      disabled={isLoading || !isRunning}
                      variant="destructive"
                      className="flex items-center gap-2"
                    >
                      <Square className="h-4 w-4" />
                      Stop Simulation
                    </Button>
                    <Button
                      onClick={checkSimulationStatus}
                      disabled={isLoading}
                      variant="outline"
                      className="flex items-center gap-2"
                    >
                      <RefreshCw className="h-4 w-4" />
                      Refresh Status
                    </Button>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">Status:</span>
                    <Badge variant={isRunning ? "default" : "secondary"}>
                      {isRunning ? "Running" : "Stopped"}
                    </Badge>
                  </div>
                  
                  {simulationStatus && (
                    <div className="text-sm text-gray-600">
                      {simulationStatus.message}
                    </div>
                  )}
                  
                  <div className="text-xs text-gray-500">
                    Debug: isRunning={isRunning?.toString() || "undefined"}, message="{simulationStatus?.message}"
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Simulation View</CardTitle>
                </CardHeader>
                <CardContent>
                  {isRunning ? (
                    <div className="text-center py-8">
                      <div className="text-lg font-medium text-green-600 mb-2">
                        ✓ MuJoCo Simulation is Running
                      </div>
                      <p className="text-gray-600">
                        The MuJoCo viewer window should have opened automatically. 
                        You can interact with the 3D simulation view there.
                      </p>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <div className="text-lg font-medium text-gray-600 mb-2">
                        Simulation Not Running
                      </div>
                      <p className="text-gray-500">
                        Click "Launch Simulation" to start the MuJoCo simulation.
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* -------------- Kinematics Tab -------------- */}
          <TabsContent value="kinematics">
            <div className="grid gap-6">
              {/* API Key Configuration Card */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Key className="h-5 w-5" />
                    API Configuration
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm font-medium">Anthropic API Key Status:</span>
                    <Badge variant={apiKeySet ? "default" : "secondary"}>
                      {apiKeySet ? "✓ Set" : "Not Set"}
                    </Badge>
                  </div>
                  
                  {!apiKeySet && (
                    <div className="space-y-3">
                      <div className="space-y-2">
                        <Label htmlFor="api-key" className="text-sm font-medium">
                          Anthropic API Key
                        </Label>
                        <div className="flex gap-2">
                          <div className="relative flex-1">
                            <Input
                              id="api-key"
                              type={showApiKey ? "text" : "password"}
                              placeholder="sk-ant-..."
                              value={apiKey}
                              onChange={(e) => setApiKey(e.target.value)}
                              className="pr-10"
                            />
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                              onClick={() => setShowApiKey(!showApiKey)}
                            >
                              {showApiKey ? (
                                <EyeOff className="h-4 w-4" />
                              ) : (
                                <Eye className="h-4 w-4" />
                              )}
                            </Button>
                          </div>
                          <Button
                            onClick={setAnthropicApiKey}
                            disabled={isSettingApiKey || !apiKey.trim()}
                            className="flex items-center gap-2"
                          >
                            <Key className="h-4 w-4" />
                            {isSettingApiKey ? "Setting..." : "Set Key"}
                          </Button>
                        </div>
                      </div>
                      <p className="text-xs text-gray-500">
                        Required for AI vision features. Get your API key from{" "}
                        <a 
                          href="https://console.anthropic.com/" 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline"
                        >
                          Anthropic Console
                        </a>
                      </p>
                    </div>
                  )}
                  
                  {apiKeySet && (
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-green-600">✓ API key is configured and ready for AI features</span>
                      <Button
                        onClick={() => setApiKeySet(false)}
                        variant="outline"
                        size="sm"
                      >
                        Update Key
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <RefreshCw className="h-5 w-5" />
                    Kinematics Control
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {kinematicsError && (
                    <div className="p-3 bg-red-100 border border-red-400 text-red-700 rounded">
                      Error: {kinematicsError}
                    </div>
                  )}

                  {/* AI Task Input */}
                  <div className="space-y-2">
                    <Label htmlFor="ai-task" className="text-sm font-medium">
                      AI Task Description (for AI Target mode)
                    </Label>
                    <Input
                      id="ai-task"
                      type="text"
                      placeholder="e.g., red cup, blue bottle, green apple..."
                      value={aiTask}
                      onChange={(e) => setAiTask(e.target.value)}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500">
                      Describe the object you want the AI to locate and track
                    </p>
                  </div>

                  <div className="flex items-center gap-4 flex-wrap">
                    <Button
                      onClick={() => launchKinematics("manual")}
                      disabled={isKinematicsLoading || isKinematicsRunning}
                      className="flex items-center gap-2"
                    >
                      <Play className="h-4 w-4" />
                      Manual Target
                    </Button>
                    <Button
                      onClick={() => launchKinematics("ai")}
                      disabled={isKinematicsLoading || isKinematicsRunning || !apiKeySet}
                      className="flex items-center gap-2"
                    >
                      <Play className="h-4 w-4" />
                      AI Target {!apiKeySet && "(Requires API Key)"}
                    </Button>
                    <Button
                      onClick={stopKinematics}
                      disabled={isKinematicsLoading || !isKinematicsRunning}
                      variant="destructive"
                      className="flex items-center gap-2"
                    >
                      <Square className="h-4 w-4" />
                      Stop
                    </Button>
                    <Button
                      onClick={checkKinematicsStatus}
                      disabled={isKinematicsLoading}
                      variant="outline"
                      className="flex items-center gap-2"
                    >
                      <RefreshCw className="h-4 w-4" />
                      Refresh Status
                    </Button>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">Status:</span>
                    <Badge variant={isKinematicsRunning ? "default" : "secondary"}>
                      {isKinematicsRunning ? "Running" : "Stopped"}
                    </Badge>
                  </div>

                  {kinematicsStatus && (
                    <div className="text-sm text-gray-600">
                      {kinematicsStatus.message}
                    </div>
                  )}

                  <div className="text-xs text-gray-500">
                    Debug: isRunning={isKinematicsRunning?.toString() || "undefined"}, message="{kinematicsStatus?.message}"
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default SimulationPage; 
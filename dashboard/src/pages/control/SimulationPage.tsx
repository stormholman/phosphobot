import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Play, Square, RefreshCw } from "lucide-react";
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
    setIsKinematicsLoading(true);
    setKinematicsError(null);
    try {
      const url = `http://localhost:80/kinematics/launch?mode=${mode}`;
      const response = await fetch(url, {
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
                      disabled={isKinematicsLoading || isKinematicsRunning}
                      className="flex items-center gap-2"
                    >
                      <Play className="h-4 w-4" />
                      AI Target
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
import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const SimulationPage: React.FC = () => {
  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <header className="bg-white shadow-md p-4">
        <h1 className="text-2xl font-bold">Simulation</h1>
      </header>
      <main className="flex-grow p-4">
        <Card>
          <CardHeader>
            <CardTitle>Simulation View</CardTitle>
          </CardHeader>
          <CardContent>
            <p>
              This page will display the simulation view.
            </p>
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default SimulationPage; 
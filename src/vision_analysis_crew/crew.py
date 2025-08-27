from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from .tools.vision_tool import VisionAnalysisTool


@CrewBase
class VisionAnalysisCrew:
    """視覺分析 Crew - 專門進行圖片分析和報告生成"""

    @agent
    def vision_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["vision_analyst"],
            verbose=True,
            tools=[VisionAnalysisTool()],
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["report_writer"],
            verbose=True,
        )

    @task
    def vision_analysis_task(self) -> Task:
        return Task(config=self.tasks_config["vision_analysis_task"], tools=[VisionAnalysisTool()])

    @task
    def report_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config["report_generation_task"],
            output_file="output/vision_analysis_report.md",
        )

    @crew
    def crew(self) -> Crew:
        """建立視覺分析 Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            # output_log_file=True,
        )

#!/usr/bin/env python3
"""
Static CLI Comparison Analysis

Analyze both CLI implementations without running them by examining
code structure, imports, method signatures, and architectural patterns.
"""

import os
import sys
import ast
import time
import json
from typing import Dict, List, Set, Any

def analyze_file_structure(filepath: str) -> Dict[str, Any]:
    """
    Analyze the structure of a Python file using AST parsing.
    
    Args:
        filepath: Path to the Python file to analyze
        
    Returns:
        Dictionary containing structural analysis
    """
    try:
        with open(filepath, 'r') as f:
            source_code = f.read()
        
        tree = ast.parse(source_code)
        
        analysis = {
            "file_size": len(source_code),
            "line_count": len(source_code.split('\n')),
            "imports": [],
            "classes": [],
            "functions": [],
            "methods": {},
            "docstrings": 0,
            "comments": source_code.count('#'),
            "complexity_score": 0
        }
        
        for node in ast.walk(tree):
            # Analyze imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    analysis["imports"].append(f"{module}.{alias.name}")
            
            # Analyze classes
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "methods": [],
                    "docstring": ast.get_docstring(node) is not None,
                    "line_number": node.lineno
                }
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            "name": item.name,
                            "args": len(item.args.args),
                            "docstring": ast.get_docstring(item) is not None,
                            "line_number": item.lineno,
                            "is_private": item.name.startswith('_'),
                            "async": isinstance(item, ast.AsyncFunctionDef)
                        }
                        class_info["methods"].append(method_info)
                
                analysis["classes"].append(class_info)
                analysis["methods"][node.name] = [m["name"] for m in class_info["methods"]]
            
            # Analyze standalone functions
            elif isinstance(node, ast.FunctionDef) and not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                func_info = {
                    "name": node.name,
                    "args": len(node.args.args),
                    "docstring": ast.get_docstring(node) is not None,
                    "line_number": node.lineno
                }
                analysis["functions"].append(func_info)
            
            # Count docstrings
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if ast.get_docstring(node):
                    analysis["docstrings"] += 1
        
        # Calculate complexity score (simple metric)
        analysis["complexity_score"] = (
            len(analysis["classes"]) * 2 +
            len(analysis["functions"]) +
            sum(len(methods) for methods in analysis["methods"].values()) +
            analysis["line_count"] // 100
        )
        
        return analysis
        
    except Exception as e:
        return {"error": str(e)}

def compare_architectural_patterns(original_analysis: Dict, openai_analysis: Dict) -> Dict[str, Any]:
    """
    Compare architectural patterns between the two CLI implementations.
    
    Args:
        original_analysis: Analysis results from original CLI
        openai_analysis: Analysis results from OpenAI SDK CLI
        
    Returns:
        Dictionary containing architectural comparison
    """
    comparison = {
        "size_comparison": {},
        "import_comparison": {},
        "class_comparison": {},
        "method_comparison": {},
        "complexity_comparison": {},
        "documentation_comparison": {},
        "architectural_differences": []
    }
    
    # Size comparison
    comparison["size_comparison"] = {
        "original_lines": original_analysis["line_count"],
        "openai_lines": openai_analysis["line_count"],
        "size_difference": openai_analysis["line_count"] - original_analysis["line_count"],
        "winner": "OpenAI CLI" if openai_analysis["line_count"] > original_analysis["line_count"] else "Original CLI"
    }
    
    # Import comparison
    original_imports = set(original_analysis["imports"])
    openai_imports = set(openai_analysis["imports"])
    
    comparison["import_comparison"] = {
        "original_count": len(original_imports),
        "openai_count": len(openai_imports),
        "common_imports": list(original_imports & openai_imports),
        "original_only": list(original_imports - openai_imports),
        "openai_only": list(openai_imports - original_imports)
    }
    
    # Class comparison
    original_classes = [c["name"] for c in original_analysis["classes"]]
    openai_classes = [c["name"] for c in openai_analysis["classes"]]
    
    comparison["class_comparison"] = {
        "original_classes": original_classes,
        "openai_classes": openai_classes,
        "common_classes": list(set(original_classes) & set(openai_classes)),
        "different_classes": list(set(original_classes) ^ set(openai_classes))
    }
    
    # Method comparison (for main classes)
    main_original_class = original_analysis["classes"][0] if original_analysis["classes"] else None
    main_openai_class = openai_analysis["classes"][0] if openai_analysis["classes"] else None
    
    if main_original_class and main_openai_class:
        original_methods = set(m["name"] for m in main_original_class["methods"])
        openai_methods = set(m["name"] for m in main_openai_class["methods"])
        
        comparison["method_comparison"] = {
            "original_methods": list(original_methods),
            "openai_methods": list(openai_methods),
            "common_methods": list(original_methods & openai_methods),
            "original_only": list(original_methods - openai_methods),
            "openai_only": list(openai_methods - original_methods),
            "method_compatibility": len(original_methods & openai_methods) / max(len(original_methods), len(openai_methods)) * 100
        }
    
    # Complexity comparison
    comparison["complexity_comparison"] = {
        "original_complexity": original_analysis["complexity_score"],
        "openai_complexity": openai_analysis["complexity_score"],
        "complexity_difference": openai_analysis["complexity_score"] - original_analysis["complexity_score"],
        "simpler": "OpenAI CLI" if openai_analysis["complexity_score"] < original_analysis["complexity_score"] else "Original CLI"
    }
    
    # Documentation comparison
    comparison["documentation_comparison"] = {
        "original_docstrings": original_analysis["docstrings"],
        "openai_docstrings": openai_analysis["docstrings"],
        "original_comments": original_analysis["comments"],
        "openai_comments": openai_analysis["comments"],
        "better_documented": "OpenAI CLI" if openai_analysis["docstrings"] > original_analysis["docstrings"] else "Original CLI"
    }
    
    # Identify architectural differences
    differences = []
    
    if "langchain" in str(original_analysis["imports"]).lower():
        differences.append("Original uses LangChain framework")
    if "openai" in str(openai_analysis["imports"]).lower() and "chatopen" not in str(openai_analysis["imports"]).lower():
        differences.append("OpenAI uses direct OpenAI SDK")
    if len(comparison["import_comparison"]["original_only"]) > 5:
        differences.append("Original has significantly more dependencies")
    if len(comparison["import_comparison"]["openai_only"]) > 5:
        differences.append("OpenAI has significantly more dependencies")
    if abs(comparison["complexity_comparison"]["complexity_difference"]) > 10:
        differences.append("Significant complexity difference between implementations")
    
    comparison["architectural_differences"] = differences
    
    return comparison

def analyze_code_quality(analysis: Dict) -> Dict[str, Any]:
    """
    Analyze code quality metrics from the structural analysis.
    
    Args:
        analysis: Structural analysis dictionary
        
    Returns:
        Dictionary containing code quality metrics
    """
    quality_metrics = {
        "documentation_ratio": 0,
        "method_to_class_ratio": 0,
        "comment_density": 0,
        "average_method_size": 0,
        "code_organization_score": 0,
        "quality_grade": "Unknown"
    }
    
    if analysis.get("line_count", 0) > 0:
        quality_metrics["comment_density"] = analysis["comments"] / analysis["line_count"] * 100
    
    if analysis.get("classes"):
        total_methods = sum(len(c["methods"]) for c in analysis["classes"])
        if len(analysis["classes"]) > 0:
            quality_metrics["method_to_class_ratio"] = total_methods / len(analysis["classes"])
        
        total_documented_methods = sum(
            sum(1 for m in c["methods"] if m["docstring"]) 
            for c in analysis["classes"]
        )
        if total_methods > 0:
            quality_metrics["documentation_ratio"] = total_documented_methods / total_methods * 100
    
    # Calculate organization score
    org_score = 0
    if analysis.get("docstrings", 0) > 0:
        org_score += 20
    if quality_metrics["comment_density"] > 5:
        org_score += 20
    if quality_metrics["documentation_ratio"] > 50:
        org_score += 30
    if len(analysis.get("classes", [])) > 0:
        org_score += 20
    if analysis.get("complexity_score", 0) < 50:
        org_score += 10
    
    quality_metrics["code_organization_score"] = org_score
    
    # Assign quality grade
    if org_score >= 80:
        quality_metrics["quality_grade"] = "A"
    elif org_score >= 60:
        quality_metrics["quality_grade"] = "B"
    elif org_score >= 40:
        quality_metrics["quality_grade"] = "C"
    elif org_score >= 20:
        quality_metrics["quality_grade"] = "D"
    else:
        quality_metrics["quality_grade"] = "F"
    
    return quality_metrics

def generate_static_report(original_analysis: Dict, openai_analysis: Dict, comparison: Dict) -> str:
    """
    Generate a comprehensive static analysis report.
    
    Args:
        original_analysis: Analysis of original CLI
        openai_analysis: Analysis of OpenAI SDK CLI
        comparison: Comparison between the two
        
    Returns:
        Formatted report string
    """
    # Calculate quality metrics
    original_quality = analyze_code_quality(original_analysis)
    openai_quality = analyze_code_quality(openai_analysis)
    
    report = f"""
{'='*80}
                CLI STATIC ANALYSIS REPORT
{'='*80}

FILE SIZE COMPARISON:
                    Original CLI    OpenAI SDK CLI    Winner
Lines of Code:      {original_analysis['line_count']:,}             {openai_analysis['line_count']:,}             {comparison['size_comparison']['winner']}
File Size (chars):  {original_analysis['file_size']:,}            {openai_analysis['file_size']:,}            {'OpenAI CLI' if openai_analysis['file_size'] > original_analysis['file_size'] else 'Original CLI'}
Classes:            {len(original_analysis['classes'])}                {len(openai_analysis['classes'])}                {'OpenAI CLI' if len(openai_analysis['classes']) > len(original_analysis['classes']) else 'Original CLI'}
Functions:          {len(original_analysis['functions'])}                {len(openai_analysis['functions'])}                {'OpenAI CLI' if len(openai_analysis['functions']) > len(original_analysis['functions']) else 'Original CLI'}

ARCHITECTURAL ANALYSIS:
                    Original CLI    OpenAI SDK CLI
Total Imports:      {comparison['import_comparison']['original_count']}                {comparison['import_comparison']['openai_count']}
Complexity Score:   {original_analysis['complexity_score']}                {openai_analysis['complexity_score']}
Docstrings:         {original_analysis['docstrings']}                {openai_analysis['docstrings']}
Comments:           {original_analysis['comments']}                {openai_analysis['comments']}

CODE QUALITY METRICS:
                        Original CLI    OpenAI SDK CLI    Winner
Documentation Ratio:    {original_quality['documentation_ratio']:.1f}%           {openai_quality['documentation_ratio']:.1f}%           {openai_quality['quality_grade'] if openai_quality['documentation_ratio'] > original_quality['documentation_ratio'] else original_quality['quality_grade']}
Comment Density:        {original_quality['comment_density']:.1f}%           {openai_quality['comment_density']:.1f}%           {'OpenAI CLI' if openai_quality['comment_density'] > original_quality['comment_density'] else 'Original CLI'}
Organization Score:     {original_quality['code_organization_score']}/100          {openai_quality['code_organization_score']}/100          {'OpenAI CLI' if openai_quality['code_organization_score'] > original_quality['code_organization_score'] else 'Original CLI'}
Quality Grade:          {original_quality['quality_grade']}                {openai_quality['quality_grade']}                {'OpenAI CLI' if openai_quality['quality_grade'] < original_quality['quality_grade'] else 'Original CLI'}

DEPENDENCY ANALYSIS:
Common Imports: {len(comparison['import_comparison']['common_imports'])}
Original Only:  {len(comparison['import_comparison']['original_only'])} imports
OpenAI Only:    {len(comparison['import_comparison']['openai_only'])} imports

Original Unique Dependencies:
{chr(10).join(f'  • {imp}' for imp in comparison['import_comparison']['original_only'][:10])}

OpenAI Unique Dependencies:  
{chr(10).join(f'  • {imp}' for imp in comparison['import_comparison']['openai_only'][:10])}

METHOD COMPATIBILITY:
"""
    
    if comparison.get('method_comparison'):
        report += f"""Common Methods:     {len(comparison['method_comparison']['common_methods'])}
Original Only:      {len(comparison['method_comparison']['original_only'])}
OpenAI Only:        {len(comparison['method_comparison']['openai_only'])}
Compatibility:      {comparison['method_comparison']['method_compatibility']:.1f}%
"""
    
    report += f"""
ARCHITECTURAL DIFFERENCES:
{chr(10).join(f'• {diff}' for diff in comparison['architectural_differences']) if comparison['architectural_differences'] else '• No significant architectural differences detected'}

PERFORMANCE PREDICTIONS:
• Initialization: {'OpenAI CLI likely faster (fewer dependencies)' if len(comparison['import_comparison']['openai_only']) < len(comparison['import_comparison']['original_only']) else 'Original CLI likely faster'}
• Memory Usage: {'OpenAI CLI likely lower' if openai_analysis['complexity_score'] < original_analysis['complexity_score'] else 'Original CLI likely lower'}
• Maintainability: {'OpenAI CLI' if openai_quality['quality_grade'] <= original_quality['quality_grade'] else 'Original CLI'} (based on documentation and organization)

RECOMMENDATIONS:
"""
    
    # Add specific recommendations
    if openai_quality['documentation_ratio'] > original_quality['documentation_ratio']:
        report += "• OpenAI CLI shows better documentation practices\n"
    if openai_analysis['complexity_score'] < original_analysis['complexity_score']:
        report += "• OpenAI CLI has lower complexity and may be easier to maintain\n"
    if len(comparison['import_comparison']['openai_only']) < len(comparison['import_comparison']['original_only']):
        report += "• OpenAI CLI has fewer dependencies, potentially faster startup\n"
    if comparison.get('method_comparison', {}).get('method_compatibility', 0) > 80:
        report += "• High method compatibility suggests easy migration between implementations\n"
    
    report += f"\n{'='*80}\n"
    
    return report

def main():
    """
    Main function to run static CLI analysis.
    """
    print("🔍 CLI Static Analysis")
    print("=" * 40)
    
    # File paths
    cli_path = "cli.py"
    cli_new_path = "cli_new.py"
    
    # Check if files exist
    if not os.path.exists(cli_path):
        print(f"❌ Original CLI file not found: {cli_path}")
        return
    
    if not os.path.exists(cli_new_path):
        print(f"❌ OpenAI SDK CLI file not found: {cli_new_path}")
        return
    
    print("📊 Analyzing file structures...")
    
    # Analyze both files
    original_analysis = analyze_file_structure(cli_path)
    openai_analysis = analyze_file_structure(cli_new_path)
    
    if "error" in original_analysis:
        print(f"❌ Error analyzing original CLI: {original_analysis['error']}")
        return
    
    if "error" in openai_analysis:
        print(f"❌ Error analyzing OpenAI CLI: {openai_analysis['error']}")
        return
    
    print("🔄 Comparing architectures...")
    
    # Compare architectures
    comparison = compare_architectural_patterns(original_analysis, openai_analysis)
    
    # Generate report
    report = generate_static_report(original_analysis, openai_analysis, comparison)
    print(report)
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"static_cli_analysis_{timestamp}.json"
    
    results = {
        "timestamp": timestamp,
        "original_analysis": original_analysis,
        "openai_analysis": openai_analysis,
        "comparison": comparison
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📄 Detailed analysis saved to: {filename}")

if __name__ == "__main__":
    main()
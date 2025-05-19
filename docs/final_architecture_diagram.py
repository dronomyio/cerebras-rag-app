#!/usr/bin/env python3
"""
Final Architecture Diagram for Cerebras RAG Application with Configurable Unstructured.io Integration
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

# Set up the figure
plt.figure(figsize=(16, 12))
ax = plt.gca()
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)

# Remove axes
ax.axis('off')

# Define colors
COLORS = {
    'user': '#E1F5FE',
    'nginx': '#B3E5FC',
    'webapp': '#81D4FA',
    'redis': '#4FC3F7',
    'weaviate': '#29B6F6',
    't2v': '#03A9F4',
    'pdf': '#039BE5',
    'unstructured': '#0288D1',
    'code': '#01579B',
    'cerebras': '#0277BD',
    'docker': '#ECEFF1',
    'arrow': '#546E7A',
    'text': '#263238',
    'border': '#90A4AE',
    'plugin': '#80DEEA',
    'config': '#FFE0B2'
}

# Helper function to draw a box with title
def draw_box(x, y, width, height, title, color, alpha=0.8, fontsize=10):
    rect = patches.Rectangle((x, y), width, height, linewidth=1, 
                            edgecolor=COLORS['border'], facecolor=color, 
                            alpha=alpha, zorder=1)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height - 0.3, title, 
            horizontalalignment='center', fontsize=fontsize, 
            fontweight='bold', color=COLORS['text'])
    return rect

# Helper function to draw an arrow
def draw_arrow(start, end, text=None, offset=0, color=COLORS['arrow'], 
               connectionstyle="arc3,rad=0.1", text_offset=0):
    arrow = patches.FancyArrowPatch(
        start, end, 
        arrowstyle='-|>', linewidth=1.5,
        color=color, connectionstyle=connectionstyle,
        zorder=2
    )
    ax.add_patch(arrow)
    
    if text:
        # Calculate the midpoint of the arrow with offset
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Add offset perpendicular to the arrow direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Normalize and rotate by 90 degrees for perpendicular offset
            nx = -dy / length
            ny = dx / length
            
            mid_x += offset * nx
            mid_y += offset * ny
            
        ax.text(mid_x, mid_y + text_offset, text, 
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=8, color=COLORS['text'],
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# Draw Docker Compose environment box
docker_box = draw_box(0.5, 0.5, 15, 11, "Docker Compose Environment", 
                     COLORS['docker'], alpha=0.3, fontsize=14)

# Draw User
user_box = draw_box(1, 10, 2, 1, "User", COLORS['user'])

# Draw NGINX
nginx_box = draw_box(4, 10, 2, 1, "NGINX", COLORS['nginx'])

# Draw Web Application
webapp_box = draw_box(4, 8, 2, 1, "Web Application", COLORS['webapp'])

# Draw Redis
redis_box = draw_box(1, 8, 2, 1, "Redis", COLORS['redis'])

# Draw Weaviate
weaviate_box = draw_box(4, 6, 2, 1, "Weaviate", COLORS['weaviate'])

# Draw Text2Vec Transformers
t2v_box = draw_box(1, 6, 2, 1, "Text2Vec\nTransformers", COLORS['t2v'])

# Draw Document Processor Service
doc_processor_box = draw_box(4, 4, 2, 1, "Document Processor\nService", COLORS['unstructured'])

# Draw Configuration
config_box = draw_box(2.5, 3, 1, 0.6, "config.yaml", COLORS['config'])

# Draw Unstructured.io API
unstructured_box = draw_box(1, 4, 2, 1, "Unstructured.io\nAPI", COLORS['unstructured'])

# Draw Plugin System
plugin_box = draw_box(1, 2, 2, 1, "Processor\nPlugins", COLORS['plugin'])

# Draw File System
file_box = draw_box(4, 2, 2, 1, "File System\n(Documents)", COLORS['user'])

# Draw Code Executor
code_box = draw_box(7, 8, 2, 1, "Code Executor", COLORS['code'])

# Draw Cerebras API (external)
cerebras_box = draw_box(10, 6, 2, 1, "Cerebras API", COLORS['cerebras'])

# Draw connections
# User to NGINX
draw_arrow((3, 10.5), (4, 10.5), "HTTPS", offset=0.2)
draw_arrow((4, 10.3), (3, 10.3), "Response", offset=-0.2)

# NGINX to Web App
draw_arrow((5, 10), (5, 9), "Proxy")

# Web App to Redis
draw_arrow((4, 8.5), (3, 8.5), "Session/Cache", offset=0.2)
draw_arrow((3, 8.3), (4, 8.3), "Data", offset=-0.2)

# Web App to Code Executor
draw_arrow((6, 8.5), (7, 8.5), "Execute Code", offset=0.2)
draw_arrow((7, 8.3), (6, 8.3), "Results", offset=-0.2)

# Web App to Weaviate
draw_arrow((5, 8), (5, 7), "Query")
draw_arrow((4.8, 7), (4.8, 8), "Results")

# Web App to Document Processor
draw_arrow((4.3, 8), (4.3, 5), "Upload Files", connectionstyle="arc3,rad=-0.2")

# Web App to Cerebras
draw_arrow((6, 8.2), (10, 6.5), "LLM Query", 
          connectionstyle="arc3,rad=-0.1")
draw_arrow((10, 6.3), (6, 8), "Response", 
          connectionstyle="arc3,rad=0.1")

# Weaviate to Text2Vec
draw_arrow((4, 6.5), (3, 6.5), "Vectorize", offset=0.2)
draw_arrow((3, 6.3), (4, 6.3), "Embeddings", offset=-0.2)

# Document Processor to Weaviate
draw_arrow((5, 5), (5, 6), "Ingest Chunks")

# Document Processor to Config
draw_arrow((4, 4.3), (3.5, 3.3), "Read Config", connectionstyle="arc3,rad=-0.2")

# Document Processor to Unstructured.io (conditional)
draw_arrow((4, 4.5), (3, 4.5), "Process Files\n(if enabled)", offset=0.2)
draw_arrow((3, 4.3), (4, 4.3), "Structured Elements", offset=-0.2)

# Document Processor to Plugins
draw_arrow((3, 4), (2, 3), "Load Plugins", connectionstyle="arc3,rad=0.2")

# Document Processor to File System
draw_arrow((4, 4.5), (4.5, 3), "Read Files", connectionstyle="arc3,rad=-0.2")

# Add title
plt.text(8, 11.5, "Cerebras RAG Architecture with Configurable Unstructured.io Integration", 
         horizontalalignment='center', fontsize=16, 
         fontweight='bold', color=COLORS['text'])

# Add legend for data flow
legend_x = 12
legend_y = 10
ax.text(legend_x, legend_y + 0.5, "Data Flow Legend:", 
        fontsize=10, fontweight='bold', color=COLORS['text'])

# Legend items
legend_items = [
    ("User Queries", COLORS['arrow']),
    ("Data Processing", COLORS['weaviate']),
    ("Document Processing", COLORS['unstructured']),
    ("LLM Inference", COLORS['cerebras']),
    ("Plugin System", COLORS['plugin']),
    ("Configuration", COLORS['config'])
]

for i, (label, color) in enumerate(legend_items):
    y_pos = legend_y - i * 0.4
    ax.add_patch(patches.Rectangle((legend_x, y_pos), 0.3, 0.2, 
                                  facecolor=color, edgecolor='none'))
    ax.text(legend_x + 0.4, y_pos + 0.1, label, fontsize=8, 
            verticalalignment='center', color=COLORS['text'])

# Add plugin details
plugin_x = 7
plugin_y = 3
plugin_width = 5
plugin_height = 2.5
plugin_detail = patches.Rectangle((plugin_x, plugin_y), plugin_width, plugin_height, 
                                 linewidth=1, edgecolor=COLORS['border'], 
                                 facecolor='white', alpha=0.9, zorder=1)
ax.add_patch(plugin_detail)
ax.text(plugin_x + plugin_width/2, plugin_y + plugin_height - 0.3, "Pluggable Document Processor System", 
        horizontalalignment='center', fontsize=10, 
        fontweight='bold', color=COLORS['text'])

# Draw plugin modules
plugin_modules = [
    ("PDF Processor", 0.5, 1.5, COLORS['pdf']),
    ("DOCX Processor", 2, 1.5, COLORS['plugin']),
    ("Text Processor", 3.5, 1.5, COLORS['plugin']),
    ("Custom Processor", 2, 0.5, COLORS['plugin'])
]

for title, x_offset, y_offset, color in plugin_modules:
    module_x = plugin_x + x_offset
    module_y = plugin_y + y_offset
    module_width = 1.5
    module_height = 0.6
    
    module = patches.Rectangle((module_x, module_y), module_width, module_height, 
                              linewidth=1, edgecolor=COLORS['border'], 
                              facecolor=color, alpha=0.8, zorder=2)
    ax.add_patch(module)
    ax.text(module_x + module_width/2, module_y + module_height/2, title, 
            horizontalalignment='center', verticalalignment='center',
            fontsize=8, fontweight='bold', color=COLORS['text'])

# Add configuration details
config_detail_x = plugin_x + 0.5
config_detail_y = plugin_y + 0.5
config_detail_width = 4
config_detail_height = 0.8

config_detail = patches.Rectangle((config_detail_x, config_detail_y), config_detail_width, config_detail_height, 
                                 linewidth=1, edgecolor=COLORS['border'], 
                                 facecolor=COLORS['config'], alpha=0.8, zorder=2)
ax.add_patch(config_detail)
ax.text(config_detail_x + 0.1, config_detail_y + config_detail_height/2, 
        "enable_unstructured_io: true/false", 
        horizontalalignment='left', verticalalignment='center',
        fontsize=8, fontweight='bold', color=COLORS['text'])

# Add annotations
annotations = [
    (13, 9, "1. User uploads multimodal documents"),
    (13, 8.5, "2. Document processor reads configuration"),
    (13, 8, "3. If Unstructured.io is enabled, it processes files"),
    (13, 7.5, "4. If disabled, only local plugins are used"),
    (13, 7, "5. Processed chunks are stored in Weaviate"),
    (13, 6.5, "6. User queries retrieve relevant chunks"),
    (13, 6, "7. Cerebras generates responses from context"),
    (13, 5.5, "8. Code examples can be executed")
]

for x, y, text in annotations:
    ax.text(x, y, text, fontsize=8, color=COLORS['text'],
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

# Save the diagram
plt.tight_layout()
plt.savefig('/home/ubuntu/cerebras-rag-app/docs/final_architecture_diagram.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("Final architecture diagram created successfully!")

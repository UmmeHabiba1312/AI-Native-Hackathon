# Data Model: AI-Native Textbook – Physical AI & Humanoid Robotics

## 1. Core Entities

### Chapter
- `id` (string): Unique identifier for the chapter
- `title` (string): Title of the chapter
- `module_id` (string): Reference to the parent module
- `content` (string): Markdown content of the chapter
- `learning_objectives` (array of strings): List of learning objectives
- `code_examples` (array of objects): Code examples in the chapter
- `diagrams` (array of objects): Diagram references in the chapter
- `exercises` (array of objects): Exercises associated with the chapter
- `metadata` (object): Additional metadata (difficulty, estimated_time, prerequisites)
- `created_at` (datetime): Timestamp of creation
- `updated_at` (datetime): Timestamp of last update
- `version` (string): Version of the chapter content

### Module
- `id` (string): Unique identifier for the module
- `title` (string): Title of the module
- `description` (string): Brief description of the module
- `chapters` (array of strings): References to chapter IDs in the module
- `order` (integer): Order of the module in the curriculum
- `created_at` (datetime): Timestamp of creation
- `updated_at` (datetime): Timestamp of last update

### User
- `id` (string): Unique identifier for the user
- `email` (string): User's email address
- `name` (string): User's full name
- `profile` (object): User profile information
  - `background` (string): Software/hardware background
  - `preferences` (object): User preferences
    - `language` (string): Preferred language (default: English)
    - `difficulty_level` (string): Preferred difficulty (beginner, intermediate, advanced)
  - `progress` (object): Learning progress tracking
- `created_at` (datetime): Timestamp of account creation
- `updated_at` (datetime): Timestamp of last update

### UserProgress
- `id` (string): Unique identifier for the progress record
- `user_id` (string): Reference to the user
- `chapter_id` (string): Reference to the chapter
- `module_id` (string): Reference to the module
- `status` (string): Status (not_started, in_progress, completed)
- `progress_percentage` (float): Percentage of completion
- `time_spent` (integer): Time spent in seconds
- `last_accessed` (datetime): Timestamp of last access
- `completed_at` (datetime): Timestamp of completion (if completed)
- `quiz_scores` (array of objects): Scores for quizzes in the chapter

### CodeExample
- `id` (string): Unique identifier for the code example
- `chapter_id` (string): Reference to the parent chapter
- `language` (string): Programming language (python, cpp, etc.)
- `platform` (string): Target platform (ros2, isaac, unity)
- `code` (string): The actual code content
- `description` (string): Description of what the code does
- `difficulty` (string): Difficulty level
- `validation_status` (string): Status of code validation
- `created_at` (datetime): Timestamp of creation

### Diagram
- `id` (string): Unique identifier for the diagram
- `chapter_id` (string): Reference to the parent chapter
- `title` (string): Title of the diagram
- `description` (string): Description of the diagram
- `file_path` (string): Path to the diagram file
- `type` (string): Type of diagram (flowchart, illustration, urdf)
- `alt_text` (string): Alternative text for accessibility
- `created_at` (datetime): Timestamp of creation

### Exercise
- `id` (string): Unique identifier for the exercise
- `chapter_id` (string): Reference to the parent chapter
- `type` (string): Type of exercise (coding, multiple_choice, essay)
- `question` (string): The exercise question
- `solution` (string): The solution or answer
- `difficulty` (string): Difficulty level
- `hints` (array of strings): Helpful hints for the exercise
- `created_at` (datetime): Timestamp of creation

## 2. Relationships

### Module ↔ Chapter
- One-to-Many: One module contains many chapters
- `module_id` in Chapter references `id` in Module

### User ↔ UserProgress
- One-to-Many: One user has many progress records
- `user_id` in UserProgress references `id` in User

### Chapter ↔ UserProgress
- One-to-Many: One chapter has many progress records from different users
- `chapter_id` in UserProgress references `id` in Chapter

### Chapter ↔ CodeExample
- One-to-Many: One chapter contains many code examples
- `chapter_id` in CodeExample references `id` in Chapter

### Chapter ↔ Diagram
- One-to-Many: One chapter contains many diagrams
- `chapter_id` in Diagram references `id` in Chapter

### Chapter ↔ Exercise
- One-to-Many: One chapter contains many exercises
- `chapter_id` in Exercise references `id` in Chapter

## 3. Vector Database Schema (Qdrant)

### Textbook Content Vectors
- `id` (string): Unique identifier matching chapter/exercise ID
- `payload` (object):
  - `chapter_id` (string): Reference to the chapter
  - `module_id` (string): Reference to the module
  - `content_type` (string): Type (chapter, exercise, code_example)
  - `text` (string): The actual text content
  - `metadata` (object): Additional metadata
- `vector` (array of floats): Embedding vector representation

## 4. Indexes & Performance Considerations

### Primary Indexes
- User.email (unique)
- Chapter.id
- Module.id
- UserProgress.user_id + UserProgress.chapter_id (composite)

### Performance Optimizations
- Chapter content caching for frequently accessed chapters
- Vector database collection for textbook content with HNSW index
- User progress aggregation for dashboard views
- Content versioning to support updates without losing progress

## 5. Data Validation Rules

### Chapter Validation
- Content must be in valid markdown format
- Learning objectives must be non-empty array
- Metadata fields (difficulty, estimated_time) are required
- Code examples must have valid syntax for their language

### User Validation
- Email must be valid and unique
- Profile information must match expected schema
- Progress tracking must reference valid chapters/modules

### Content Relationships
- All chapter references in modules must exist
- All user progress records must reference valid users and chapters
- Code examples must reference valid chapters
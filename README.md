
*WORK IN PROGRESS*

Environment

Install VS Code
Install Git
Install Python
Install PostgreSQL

Test your PostgreSQL installation
/Library/PostgreSQL/{VERSION}/bin/psql -h localhost -p 5432 -U postgres -d postgres

Type in exit and hit enter
Type in source ~/.zprofile
Add in the following values
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=model
export PGUSER=postgres
export PGPASSWORD={your_password}

Press control+o followed by control+x

Type in source ~/.zprofile and hit enter

Install requirements
python3 -m venv ~/venvs/mlx
source ~/venvs/mlx/bin/activate
pip install -U pip
pip install -r requirements.txt

Install Hugging Face CLI
pip install -U "huggingface_hub[cli]"

Create directory and download model
mkdir -p ~/Models/openai && hf download openai/gpt-oss-20b --local-dir ~/Models/openai/gpt-oss-20b

Install Xcode from App Store

Run and confirm Xcode
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
xcrun -find metal

Open Xcode, go to Settings > Components and install Metal Toolchain

Install pg vector for embedding tables
export PG_CONFIG=/Library/PostgreSQL/{VERSION}/bin/pg_config
cd /tmp
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo --preserve-env=PG_CONFIG make install

TROUBLESHOOTING
If you run into issues with pg vector, try this
export PG_CONFIG=/Library/PostgreSQL/{VERSION}/bin/pg_config
export PG_SYSROOT="$(xcrun --show-sdk-path)"

"$PG_CONFIG" --cppflags
make clean
make PG_CONFIG="$PG_CONFIG" PG_SYSROOT="$PG_SYSROOT"
sudo make install PG_CONFIG="$PG_CONFIG" PG_SYSROOT="$PG_SYSROOT"

Install Ninja
pip install -U ninja

Clone GPT OSS 20B repo and install Metal build
cd ~
git clone https://github.com/openai/gpt-oss.git
cd ~/gpt-oss
GPTOSS_BUILD_METAL=1 pip install -e ".[metal]"

Clone the repository and open it in the integrated terminal within VS Code
source .venv/bin/activate

Run the helper script to set up the schema for PostgreSQL
python3 helpers/helper_database.py

Optionally, modify config/durable_memories-Chief.txt

Start the model server
mlx_lm.server --model Models/openai/gpt-oss-20b --prompt-cache-size 0

Start the embedder server
Open the app/src/servers/ directory in the integrated terminal within VS Code
gunicorn server_embed_true:server_embed_api --bind 127.0.0.1:8081 --workers 1 --threads 4

Run the script to receive and store an output
Python3 run.py “Hello, world!”

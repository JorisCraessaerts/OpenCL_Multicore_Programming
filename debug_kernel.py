# Sla dit bestand op als: debug_kernel.py
import pyopencl as cl
import os

# Het enige doel van dit script is om de problematische kernel te compileren 
# en de foutmelding van de driver te tonen.
KERNEL_TO_DEBUG = "kernels/union_within_tile_local_memory_optimized.cl"

print(f"--- Poging tot compilatie van: {KERNEL_TO_DEBUG} ---")

# Standaard OpenCL setup
try:
    platform = cl.get_platforms()[0]
    device = platform.get_devices(cl.device_type.GPU)[0]
    context = cl.Context([device])

    # Lees de kernel code
    print(f"Bestand '{KERNEL_TO_DEBUG}' wordt gelezen...")
    with open(KERNEL_TO_DEBUG, "r") as f:
        kernel_code = f.read()
    print("Bestand succesvol gelezen.")

    # Probeer te compileren en vang de fout op.
    # De 'e' variabele in de except-blok zal de volledige compiler-log bevatten.
    print("Kernel wordt gecompileerd...")
    program = cl.Program(context, kernel_code).build()
    
    print("\n✅ SUCCESS: De kernel is succesvol gecompileerd!")
    print("Dit is onverwacht. Als je dit ziet, dan zit het probleem ergens anders.")
    print("Gevonden kernels:", program.get_info(cl.program_info.KERNEL_NAMES))


except cl.RuntimeError as e:
    print("\n❌ FOUT: De kernel kon NIET compileren.")
    print("Hieronder staat de volledige C-compilerfout van de NVIDIA-driver:")
    print("="*60)
    # De error 'e' bevat de volledige, gedetailleerde output van de compiler.
    print(e)
    print("="*60)

except FileNotFoundError:
    print(f"\n❌ FOUT: Kan het bestand niet vinden: '{KERNEL_TO_DEBUG}'")
    print("Controleer of de bestandsnaam en het pad correct zijn.")

except Exception as e:
    print(f"Een onverwachte Python-fout is opgetreden: {e}")
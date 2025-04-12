# agents/resource/tools.py
import google.cloud.compute_v1
import logging
import time
import json
import os
from typing import List, Dict, Optional
from google.api_core.exceptions import NotFound, Conflict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper: Wait for Zone Operation ---
def _wait_for_zone_operation(project_id: str, zone: str, operation_id: str):
    """Waits for a zone operation to complete and returns the result."""
    logger.info(f"Waiting for zone operation {operation_id} in {zone} to complete...")
    zone_operations_client = google.cloud.compute_v1.ZoneOperationsClient()
    while True:
        try:
            result = zone_operations_client.get(project=project_id, zone=zone, operation=operation_id)
            if result.status == google.cloud.compute_v1.Operation.Status.DONE:
                logger.info(f"Zone operation {operation_id} finished with status DONE.")
                # Check for errors within the completed operation
                if result.error:
                    error_message = _format_operation_error(result)
                    logger.error(f"Zone operation {operation_id} completed with error: {error_message}")
                return result
            time.sleep(5) # Wait before polling again
        except Exception as e:
            logger.error(f"Error waiting for zone operation {operation_id}: {e}", exc_info=True)
            # Depending on the error, you might want to retry or raise
            raise # Re-raise the exception for now

# --- Helper: Format Operation Error ---
def _format_operation_error(operation_result) -> str:
    """Formats error details from a completed operation object."""
    error_message = "Unknown error during operation."
    # Check if the error attribute and its errors list exist and are populated
    if hasattr(operation_result, 'error') and operation_result.error and hasattr(operation_result.error, 'errors') and operation_result.error.errors:
        try:
            # Format each error detail
            error_details = [f"Code: {getattr(err, 'code', 'N/A')}, Message: {getattr(err, 'message', 'N/A')}" for err in operation_result.error.errors]
            error_message = "; ".join(error_details)
        except Exception as e: # Catch potential issues during formatting
            logger.error(f"Failed to format operation error details: {e}")
            error_message = str(operation_result.error) # Fallback to string representation
    elif hasattr(operation_result, 'error') and operation_result.error:
        # Handle cases where error exists but errors list might be missing/empty
         error_message = str(operation_result.error)
    return error_message


# --- Compute Engine Tool (Create VM) ---
def create_vm_instance_func(
    project_id: Optional[str] = None,
    zone: Optional[str] = None,
    instance_name: Optional[str] = None,
    machine_type: Optional[str] = None,
    source_image: Optional[str] = None,
    disk_size_gb: Optional[str] = None,
    disk_type: Optional[str] = None,
    subnetwork: Optional[str] = None,
    service_account_email: Optional[str] = None,
    scopes: Optional[List[str]] = None,
    metadata: Optional[Dict[str,str]] = None,
    labels: Optional[Dict[str,str]] = None
) -> str:
    """Creates a Google Compute Engine VM instance, reading defaults from environment variables."""

    # Fetch values from environment or use fallbacks
    project_id = project_id or os.environ.get('GCP_PROJECT_ID', 'hostproject1-355311')
    zone = zone or os.environ.get('VM_DEFAULT_ZONE', 'us-central1-c')
    instance_name = instance_name or os.environ.get('VM_DEFAULT_INSTANCE_NAME', 'instance-via-adk')
    machine_type = machine_type or os.environ.get('VM_DEFAULT_MACHINE_TYPE', 'e2-medium')
    source_image = source_image or os.environ.get('VM_DEFAULT_SOURCE_IMAGE', 'projects/debian-cloud/global/images/debian-12-bookworm-v20250311')
    disk_size_gb = disk_size_gb or os.environ.get('VM_DEFAULT_DISK_SIZE_GB', '10')
    disk_type = disk_type or os.environ.get('VM_DEFAULT_DISK_TYPE', 'pd-balanced')
    subnetwork = subnetwork or os.environ.get('VM_DEFAULT_SUBNETWORK', 'subnet-central1')
    effective_service_account_email = service_account_email if service_account_email is not None else os.environ.get('VM_DEFAULT_SERVICE_ACCOUNT')
    scopes = scopes or ["https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/trace.append"]
    metadata = metadata or {'enable-osconfig': 'TRUE', 'enable-oslogin': 'true'}
    labels = labels or {'goog-ops-agent-policy': 'v2-x86-template-1-4-0', 'goog-ec-src': 'vm_add-adk-tool'}

    if not all([project_id, zone, instance_name]):
         return "Error creating VM: Project ID, Zone, and Instance Name are required."

    try:
        logger.info(f"Attempting to create VM. Project: {project_id}, Zone: {zone}, Name: {instance_name}")
        instance_client = google.cloud.compute_v1.InstancesClient()

        machine_type_url = f"projects/{project_id}/zones/{zone}/machineTypes/{machine_type}"
        disk_type_url = f"projects/{project_id}/zones/{zone}/diskTypes/{disk_type}"
        region = '-'.join(zone.split('-')[:2])
        subnetwork_url = f"projects/{project_id}/regions/{region}/subnetworks/{subnetwork}"
        metadata_items = [{"key": k, "value": v} for k, v in metadata.items()]

        config = {
            "name": instance_name, "machine_type": machine_type_url, "metadata": {"items": metadata_items},
            "labels": labels,
            "disks": [{"boot": True, "auto_delete": True, "device_name": instance_name, "initialize_params": {"source_image": source_image, "disk_size_gb": disk_size_gb, "disk_type": disk_type_url,}, "mode": "READ_WRITE", "type": "PERSISTENT"}],
            "network_interfaces": [{"subnetwork": subnetwork_url, "stack_type": "IPV4_ONLY", "access_configs": [{"name": "External NAT", "network_tier": "PREMIUM", "type_": "ONE_TO_ONE_NAT"}]}],
            "scheduling": {"on_host_maintenance": "MIGRATE", "automatic_restart": True, "provisioning_model": "STANDARD"},
            "deletion_protection": False, "reservation_affinity": {"consume_reservation_type": "ANY_RESERVATION"},
            "shielded_instance_config": {"enable_integrity_monitoring": True, "enable_secure_boot": False, "enable_vtpm": True},
            "service_accounts": [{"email": effective_service_account_email, "scopes": scopes}] if effective_service_account_email else [],
        }
        operation = instance_client.insert(project=project_id, zone=zone, instance_resource=config)
        result = _wait_for_zone_operation(project_id, zone, operation.name)

        if result.error:
            error_message = _format_operation_error(result)
            logger.error(f"Error during VM creation (Op {operation.name}): {error_message}")
            return f"Error creating VM: {error_message}"
        else:
            logger.info(f"VM instance {instance_name} created successfully (Op {operation.name}).")
            return f"VM instance '{instance_name}' created successfully in project '{project_id}', zone '{zone}'."

    except Conflict:
        logger.warning(f"VM instance '{instance_name}' likely already exists in project '{project_id}', zone '{zone}'.")
        return f"Error creating VM: Instance '{instance_name}' likely already exists in zone {zone}."
    except Exception as e:
        logger.error(f"Failed to create VM instance {instance_name}: {e}", exc_info=True)
        error_detail = str(e)
        if hasattr(e, 'message'): error_detail = e.message
        elif hasattr(e, 'errors'): error_detail = str(e.errors)
        return f"Failed to create VM instance {instance_name}: {error_detail}"

# --- Compute Engine Tool (Delete VM) ---
def delete_vm_instance_func(
    project_id: Optional[str] = None,
    zone: Optional[str] = None,
    instance_name: str = ""
) -> str:
    """Deletes a specified Google Compute Engine VM instance."""
    project_id = project_id or os.environ.get('GCP_PROJECT_ID', 'hostproject1-355311')
    zone = zone or os.environ.get('VM_DEFAULT_ZONE', 'us-central1-c')

    if not instance_name: return "Error deleting VM: Instance name is required."
    if not all([project_id, zone]): return "Error deleting VM: Project ID and Zone are required."

    try:
        logger.info(f"Attempting to delete VM. Project: {project_id}, Zone: {zone}, Name: {instance_name}")
        instance_client = google.cloud.compute_v1.InstancesClient()
        operation = instance_client.delete(project=project_id, zone=zone, instance=instance_name)
        result = _wait_for_zone_operation(project_id, zone, operation.name)
        if result.error:
            error_message = _format_operation_error(result)
            logger.error(f"Error during VM deletion (Op {operation.name}): {error_message}")
            if "not found" in error_message.lower():
                 return f"VM instance '{instance_name}' not found in project '{project_id}', zone '{zone}'. It might already be deleted."
            return f"Error deleting VM '{instance_name}': {error_message}"
        else:
            logger.info(f"VM instance {instance_name} deleted successfully (Op {operation.name}).")
            return f"VM instance '{instance_name}' deleted successfully from project '{project_id}', zone '{zone}'."
    except NotFound:
        logger.warning(f"VM instance '{instance_name}' not found in project '{project_id}', zone '{zone}'.")
        return f"VM instance '{instance_name}' not found. It might already be deleted."
    except Exception as e:
        logger.error(f"Failed to delete VM instance '{instance_name}': {e}", exc_info=True)
        error_detail = str(e)
        if hasattr(e, 'message'): error_detail = e.message
        elif hasattr(e, 'errors'): error_detail = str(e.errors)
        return f"Failed to delete VM instance '{instance_name}': {error_detail}"

# --- Compute Engine Tool (List VMs) ---
def list_vm_instances_func(
    project_id: Optional[str] = None,
    zone: Optional[str] = None,
    filter_expression: Optional[str] = None
) -> str:
    """Lists Google Compute Engine VM instances in a specified project and zone."""
    project_id = project_id or os.environ.get('GCP_PROJECT_ID', 'hostproject1-355311')
    zone = zone or os.environ.get('VM_DEFAULT_ZONE', 'us-central1-c')

    if not all([project_id, zone]): return "Error listing VMs: Project ID and Zone are required."

    try:
        logger.info(f"Attempting to list VMs. Project: {project_id}, Zone: {zone}, Filter: '{filter_expression}'")
        instance_client = google.cloud.compute_v1.InstancesClient()
        list_request = google.cloud.compute_v1.ListInstancesRequest(project=project_id, zone=zone, filter=filter_expression or "")
        instance_list = instance_client.list(request=list_request)
        results = []
        instance_count = 0
        for instance in instance_list:
            instance_count += 1
            details = {"name": instance.name, "status": instance.status, "machine_type": instance.machine_type.split('/')[-1], "creation_timestamp": instance.creation_timestamp }
            try:
                ext_ip = "N/A"
                if instance.network_interfaces and instance.network_interfaces[0].access_configs:
                     # Safely get nat_ip using getattr
                     ext_ip = getattr(instance.network_interfaces[0].access_configs[0], 'nat_ip', "N/A")
                details["external_ip"] = ext_ip
            except (IndexError, AttributeError): details["external_ip"] = "N/A"
            results.append(details)

        if not results:
             return f"No VM instances found in project '{project_id}', zone '{zone}' matching the filter '{filter_expression}'." if filter_expression else f"No VM instances found in project '{project_id}', zone '{zone}'."

        output_lines = [f"Found {instance_count} VM(s) in project '{project_id}', zone '{zone}':"]
        for vm in results:
            output_lines.append(f"- Name: {vm['name']}, Status: {vm['status']}, Type: {vm['machine_type']}, IP: {vm['external_ip']}, Created: {vm['creation_timestamp']}")
        return "\n".join(output_lines)

    except Exception as e:
        logger.error(f"Failed to list VM instances: {e}", exc_info=True)
        error_detail = str(e)
        if hasattr(e, 'message'): error_detail = e.message
        elif hasattr(e, 'errors'): error_detail = str(e.errors)
        return f"Failed to list VM instances in project '{project_id}', zone '{zone}': {error_detail}"

# --- Compute Engine Tool (Start VM) ---
def start_vm_instance_func(
    project_id: Optional[str] = None,
    zone: Optional[str] = None,
    instance_name: str = ""
) -> str:
    """Starts a specified Google Compute Engine VM instance."""
    project_id = project_id or os.environ.get('GCP_PROJECT_ID', 'hostproject1-355311')
    zone = zone or os.environ.get('VM_DEFAULT_ZONE', 'us-central1-c')

    if not instance_name: return "Error starting VM: Instance name is required."
    if not all([project_id, zone]): return "Error starting VM: Project ID and Zone are required."

    try:
        logger.info(f"Attempting to start VM. Project: {project_id}, Zone: {zone}, Name: {instance_name}")
        instance_client = google.cloud.compute_v1.InstancesClient()
        operation = instance_client.start(project=project_id, zone=zone, instance=instance_name)
        result = _wait_for_zone_operation(project_id, zone, operation.name)
        if result.error:
            error_message = _format_operation_error(result)
            logger.error(f"Error during VM start (Op {operation.name}): {error_message}")
            return f"Error starting VM '{instance_name}' in project '{project_id}', zone '{zone}': {error_message}"
        else:
            logger.info(f"VM instance {instance_name} started successfully (Op {operation.name}).")
            return f"VM instance '{instance_name}' started successfully in project '{project_id}', zone '{zone}'."
    except NotFound:
        logger.warning(f"VM instance '{instance_name}' not found in project '{project_id}', zone '{zone}'. Cannot start.")
        return f"VM instance '{instance_name}' not found in project '{project_id}', zone '{zone}'."
    except Exception as e:
        logger.error(f"Failed to start VM instance '{instance_name}': {e}", exc_info=True)
        error_detail = str(e)
        if "already running" in error_detail.lower():
             logger.warning(f"Attempted to start VM '{instance_name}' which is already running.")
             return f"VM instance '{instance_name}' is already running."
        if hasattr(e, 'message'): error_detail = e.message
        elif hasattr(e, 'errors'): error_detail = str(e.errors)
        return f"Failed to start VM instance '{instance_name}': {error_detail}"

# --- Compute Engine Tool (Stop VM) ---
def stop_vm_instance_func(
    project_id: Optional[str] = None,
    zone: Optional[str] = None,
    instance_name: str = ""
) -> str:
    """Stops a specified Google Compute Engine VM instance."""
    project_id = project_id or os.environ.get('GCP_PROJECT_ID', 'hostproject1-355311')
    zone = zone or os.environ.get('VM_DEFAULT_ZONE', 'us-central1-c')

    if not instance_name: return "Error stopping VM: Instance name is required."
    if not all([project_id, zone]): return "Error stopping VM: Project ID and Zone are required."

    try:
        logger.info(f"Attempting to stop VM. Project: {project_id}, Zone: {zone}, Name: {instance_name}")
        instance_client = google.cloud.compute_v1.InstancesClient()
        operation = instance_client.stop(project=project_id, zone=zone, instance=instance_name)
        result = _wait_for_zone_operation(project_id, zone, operation.name)
        if result.error:
            error_message = _format_operation_error(result)
            logger.error(f"Error during VM stop (Op {operation.name}): {error_message}")
            return f"Error stopping VM '{instance_name}' in project '{project_id}', zone '{zone}': {error_message}"
        else:
            logger.info(f"VM instance {instance_name} stopped successfully (Op {operation.name}).")
            return f"VM instance '{instance_name}' stopped successfully in project '{project_id}', zone '{zone}'."
    except NotFound:
        logger.warning(f"VM instance '{instance_name}' not found in project '{project_id}', zone '{zone}'. Cannot stop.")
        return f"VM instance '{instance_name}' not found in project '{project_id}', zone '{zone}'."
    except Exception as e:
        logger.error(f"Failed to stop VM instance '{instance_name}': {e}", exc_info=True)
        error_detail = str(e)
        if "terminated" in error_detail.lower() or "not running" in error_detail.lower():
             logger.warning(f"Attempted to stop VM '{instance_name}' which is not running.")
             return f"VM instance '{instance_name}' is already stopped (or not running)."
        if hasattr(e, 'message'): error_detail = e.message
        elif hasattr(e, 'errors'): error_detail = str(e.errors)
        return f"Failed to stop VM instance '{instance_name}': {error_detail}"

# --- Compute Engine Tool (Get Details) ---
def get_vm_instance_details_func(
    project_id: Optional[str] = None,
    zone: Optional[str] = None,
    instance_name: str = ""
) -> str:
    """Gets detailed information about a specified Google Compute Engine VM instance."""
    project_id = project_id or os.environ.get('GCP_PROJECT_ID', 'hostproject1-355311')
    zone = zone or os.environ.get('VM_DEFAULT_ZONE', 'us-central1-c')

    if not instance_name: return "Error getting VM details: Instance name is required."
    if not all([project_id, zone]): return "Error getting VM details: Project ID and Zone are required."

    try:
        logger.info(f"Attempting to get details for VM. Project: {project_id}, Zone: {zone}, Name: {instance_name}")
        instance_client = google.cloud.compute_v1.InstancesClient()
        instance = instance_client.get(project=project_id, zone=zone, instance=instance_name)

        details = {
            "name": instance.name, "id": instance.id, "status": instance.status,
            "zone": instance.zone.split('/')[-1], "machine_type": instance.machine_type.split('/')[-1],
            "cpu_platform": instance.cpu_platform, "creation_timestamp": instance.creation_timestamp,
            "labels": dict(instance.labels),
            "metadata": {item.key: item.value for item in instance.metadata.items},
            "scheduling": {"on_host_maintenance": instance.scheduling.on_host_maintenance, "automatic_restart": instance.scheduling.automatic_restart, "provisioning_model": instance.scheduling.provisioning_model},
            "network_interfaces": [], "disks": []
        }

        # Extract network details safely
        for ni in instance.network_interfaces:
            ni_details = {"name": ni.name, "network": ni.network.split('/')[-1]}
            if ni.subnetwork: ni_details["subnetwork"] = ni.subnetwork.split('/')[-1]
            if ni.network_i_p: ni_details["internal_ip"] = ni.network_i_p

            # Safely extract access config details
            if ni.access_configs:
                ni_details["access_configs"] = [
                    {
                        "name": getattr(ac, 'name', 'N/A'),
                        "type": getattr(ac, 'type_', 'N/A'), # Use type_ as type is a reserved keyword
                        "nat_ip": getattr(ac, 'nat_ip', None), # Safely get nat_ip or None
                        "network_tier": getattr(ac, 'network_tier', None) # Safely get network_tier or None
                    }
                    for ac in ni.access_configs
                ]
            details["network_interfaces"].append(ni_details)

        # Extract disk details safely
        for disk in instance.disks:
             # Ensure source is handled correctly if it might be missing or None
             disk_source = disk.source.split('/')[-1] if hasattr(disk, 'source') and disk.source else 'N/A'
             disk_details = {
                 "name": getattr(disk, 'device_name', 'N/A'),
                 "type": getattr(disk, 'type_', 'N/A'), # Use type_
                 "mode": getattr(disk, 'mode', 'N/A'),
                 "auto_delete": getattr(disk, 'auto_delete', False), # Default to False if missing
                 "boot": getattr(disk, 'boot', False), # Default to False if missing
                 "source": disk_source
             }
             details["disks"].append(disk_details)

        # Format output, safely accessing potentially None values from dict
        output = f"Details for VM '{details.get('name','N/A')}' in project '{project_id}', zone '{details.get('zone','N/A')}':\n"
        output += f"  ID: {details.get('id', 'N/A')}\n"
        output += f"  Status: {details.get('status', 'N/A')}\n"
        output += f"  Type: {details.get('machine_type', 'N/A')} ({details.get('cpu_platform', 'N/A')})\n"
        output += f"  Created: {details.get('creation_timestamp', 'N/A')}\n"

        # Network Output
        if details['network_interfaces']:
            ni = details['network_interfaces'][0] # Show first interface
            output += f"  Network: {ni.get('network', 'N/A')}, Subnet: {ni.get('subnetwork', 'N/A')}\n"
            output += f"  Internal IP: {ni.get('internal_ip', 'N/A')}\n"
            # Check if access_configs list exists and is not empty
            if ni.get('access_configs') and len(ni['access_configs']) > 0:
                ac = ni['access_configs'][0]
                ext_ip = ac.get('nat_ip')
                tier = ac.get('network_tier')
                output += f"  External IP: {ext_ip if ext_ip else 'N/A'} (Tier: {tier if tier else 'N/A'})\n"
            else:
                output += "  External IP: N/A\n" # Explicitly state N/A if no access config or IP
        else:
             output += "  Network Interfaces: N/A\n" # Indicate if no interfaces found

        # Disk Output (Show all disks)
        if details['disks']:
             output += "  Disks:\n"
             for d in details['disks']:
                  boot_indicator = "(Boot)" if d.get('boot') else ""
                  output += f"    - Name: {d.get('name','N/A')} {boot_indicator}\n"
                  output += f"      Source: {d.get('source','N/A')}\n"
                  output += f"      Type: {d.get('type','N/A')}, Mode: {d.get('mode','N/A')}, AutoDelete: {d.get('auto_delete')}\n"
        else:
             output += "  Disks: N/A\n" # Indicate if no disks found

        # Labels Output
        if details['labels']:
             output += f"  Labels: {json.dumps(details['labels'])}\n"
        else:
             output += "  Labels: None\n"

        return output.strip()

    except NotFound:
        logger.warning(f"VM instance '{instance_name}' not found in project '{project_id}', zone '{zone}'. Cannot get details.")
        return f"VM instance '{instance_name}' not found in project '{project_id}', zone '{zone}'."
    except Exception as e:
        logger.error(f"Failed to get details for VM instance '{instance_name}': {e}", exc_info=True)
        error_detail = str(e)
        if hasattr(e, 'message'): error_detail = e.message
        elif hasattr(e, 'errors'): error_detail = str(e.errors)
        return f"Failed to get details for VM instance '{instance_name}': {error_detail}"